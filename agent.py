"""Review the pull request that triggered the GitHub Actions workflow."""

import asyncio
import os
from typing import Any, Literal

import dotenv
from github import Auth, Github
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentStream,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI


dotenv.load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

if not github_token:
    raise RuntimeError("GITHUB_TOKEN is required")
if not repository:
    raise RuntimeError("REPOSITORY is required (for example, owner/repository)")
if not pr_number:
    raise RuntimeError("PR_NUMBER is required")

try:
    pull_request_number = int(pr_number)
except ValueError as error:
    raise RuntimeError("PR_NUMBER must be an integer") from error

github_client = Github(auth=Auth.Token(github_token))
repo = github_client.get_repo(repository)
review_posted = False


def get_repository_details() -> dict[str, Any]:
    """Fetch basic identifying information about the configured repository."""
    return {
        "name": repo.name,
        "full_name": repo.full_name,
        "owner": repo.owner.login,
        "description": repo.description,
        "default_branch": repo.default_branch,
        "url": repo.html_url,
    }


def get_pr_details(number: int) -> dict[str, Any]:
    """Fetch a pull request's metadata and commit SHAs.

    Args:
        number: The repository-local pull request number.
    """
    pull_request = repo.get_pull(number)
    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body or "No description provided.",
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_shas": [commit.sha for commit in pull_request.get_commits()],
    }


def get_pull_request_count(
    state: Literal["open", "closed", "all"] = "all",
) -> dict[str, int | str]:
    """Count pull requests, optionally filtered by state."""
    pull_requests = repo.get_pulls(state=state)
    return {"state": state, "count": pull_requests.totalCount}


def get_file_contents(file_path: str) -> str:
    """Fetch and decode a file from the repository's default branch.

    Args:
        file_path: Repository-relative path of the file to retrieve.
    """
    contents = repo.get_contents(file_path)
    if isinstance(contents, list):
        raise ValueError(f"{file_path!r} is a directory, not a file")
    return contents.decoded_content.decode("utf-8")


def get_pr_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Fetch changed-file details and patches for a commit SHA.

    Args:
        head_sha: Full Git commit SHA obtained from the PR details tool.
    """
    commit = repo.get_commit(head_sha)
    return [
        {
            "filename": changed_file.filename,
            "status": changed_file.status,
            "additions": changed_file.additions,
            "deletions": changed_file.deletions,
            "changes": changed_file.changes,
            "patch": changed_file.patch,
        }
        for changed_file in commit.files
    ]


async def add_context_to_state(ctx: Context, context: str) -> str:
    """Save gathered repository and pull-request context in workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["gathered_contexts"] = context
    await ctx.store.set("state", current_state)
    return "The gathered context was saved to workflow state."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save the drafted pull-request review in workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "The draft review comment was saved to workflow state."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Save the approved pull-request review in workflow state."""
    current_state = await ctx.store.get("state", default={})
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "The final review was saved to workflow state."


def post_review_to_github(number: int, comment: str) -> str:
    """Post a final review comment to a GitHub pull request."""
    global review_posted

    review = repo.get_pull(number).create_review(body=comment, event="COMMENT")
    review_posted = True
    return f"Posted review {review.id} to pull request #{number}."


repository_details_tool = FunctionTool.from_defaults(fn=get_repository_details)
pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
pr_count_tool = FunctionTool.from_defaults(fn=get_pull_request_count)
file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
pr_commit_details_tool = FunctionTool.from_defaults(fn=get_pr_commit_details)
post_review_tool = FunctionTool.from_defaults(fn=post_review_to_github)

llm_options: dict[str, Any] = {
    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "api_key": os.getenv("OPENAI_API_KEY"),
}
if openai_base_url := os.getenv("OPENAI_BASE_URL"):
    llm_options["api_base"] = openai_base_url
llm = OpenAI(**llm_options)

CONTEXT_SYSTEM_PROMPT = """You are the context gathering agent. You must gather:
- The PR author, title, body, diff URL, state, head SHA, and commit SHAs.
- The changed files and patches for every commit in the PR.
- Any repository files needed to check contribution rules, tests, migrations,
  documentation, or related source code.

Use get_pr_details with the requested PR number, then get_pr_commit_details for
every returned commit SHA. When you have enough information, summarize it, call
add_context_to_state with the complete summary, and hand off to CommentorAgent.
"""

COMMENTOR_SYSTEM_PROMPT = """You are the commentor agent. Write a human-quality
pull-request review in Markdown. Request missing context from ContextAgent.

The review must be 200-300 words, address the author directly, identify what is
good, check every contribution rule, discuss tests, discuss migrations for any
new models, discuss documentation for new endpoints, and quote specific lines
that could be improved with concrete suggestions.

If gathered_contexts is empty, hand off to ContextAgent. Otherwise draft the
complete review, call add_comment_to_state with it, and hand off to
ReviewAndPostingAgent. If that agent requests changes, revise the review, save
the new draft, and hand it back for another check.
"""

REVIEW_AND_POSTING_SYSTEM_PROMPT = """You are the review and posting agent.
Start by handing off to CommentorAgent so it can create a review. Inspect the
saved review_comment and verify every requirement: 200-300 words, strengths,
all contribution rules, test coverage, migrations for new models, endpoint
documentation, and quoted source lines with concrete improvements. A file and
line reference alone is not a quote.

If anything is missing, hand off to CommentorAgent with exact rewrite requests.
When the draft is adequate, call add_final_review_to_state with the exact review,
then call post_review_to_github with the requested PR number and the same review.
Only report success after both calls succeed.
"""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers PR details, diffs, and supporting repository files.",
    tools=[
        repository_details_tool,
        pr_details_tool,
        pr_count_tool,
        file_contents_tool,
        pr_commit_details_tool,
        add_context_to_state,
    ],
    system_prompt=CONTEXT_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Drafts a complete pull-request review from gathered context.",
    tools=[add_comment_to_state],
    system_prompt=COMMENTOR_SYSTEM_PROMPT,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Checks, approves, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt=REVIEW_AND_POSTING_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)


async def main() -> None:
    """Run the automated review workflow for the PR from the environment."""
    query = f"Write a review for PR number {pull_request_number}."
    prompt = RichPromptTemplate(query)
    handler = workflow_agent.run(user_msg=prompt.format())

    current_agent = None
    streamed_since_output = False
    printed_text = ""
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")

        if isinstance(event, AgentStream) and event.delta:
            print(event.delta, end="", flush=True)
            printed_text += event.delta
            streamed_since_output = True
        elif isinstance(event, AgentOutput):
            if event.tool_calls:
                print("Selected tools:", [call.tool_name for call in event.tool_calls])
            if streamed_since_output:
                print()
            else:
                event_content = event.response.content or ""
                if event_content:
                    print(event_content)
                    printed_text += event_content
            streamed_since_output = False
        elif isinstance(event, ToolCall):
            print(
                f"Calling selected tool: {event.tool_name}, "
                f"with arguments: {event.tool_kwargs}"
            )
        elif isinstance(event, ToolCallResult):
            if streamed_since_output:
                print()
                streamed_since_output = False
            print(f"Output from tool: {event.tool_output}")

    if streamed_since_output:
        print()

    final_result = await handler
    final_content = getattr(getattr(final_result, "response", None), "content", None)
    if final_content and final_content not in printed_text:
        print(final_content)
    elif not printed_text.strip():
        print(str(final_result).strip())

    # LLM tool selection is not deterministic. If the workflow produced a review
    # but stopped before calling the posting tool, post that review here so a
    # successful Actions run cannot silently finish without reviewing the PR.
    if not review_posted:
        review_text = (final_content or printed_text or str(final_result)).strip()
        if not review_text:
            raise RuntimeError("The workflow completed without producing a review")
        print(post_review_to_github(pull_request_number, review_text))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        github_client.close()
