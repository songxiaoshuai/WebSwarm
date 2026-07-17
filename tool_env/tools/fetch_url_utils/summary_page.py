"""
Webpage-summary tool that extracts useful information based on the original question and current fetch goal.

Note: system_prompt / system_prompt_with_link below are runtime prompts. Comment cleanup
in this file must not modify their text.
"""
from llm_infer.llm_infer import llm_infer
from datetime import datetime


# System prompt for LLM webpage summarization
system_prompt = """**Task Instruction:**

🕐 **CRITICAL TIME CONTEXT**: Today is {current_date}. When analyzing and summarizing web content, prioritize information that reflects the current date and time, especially for time-sensitive topics.

You are tasked with reading and analyzing web pages based on the following inputs: **Search Goal** and **Fetched Web Pages**. Your objective is to extract relevant and helpful information based on **Search Goal** from the **Fetched Web Pages** and seamlessly integrate this information into the **Search Goal** to continue the reasoning process for answering the original question.
Only use information explicitly mentioned on a webpage, even if the information is incomplete. You are prohibited from using your own knowledge to speculate or expand upon content that does not exist on the webpage.

**Guidelines:**

1. **Analyze the Fetched Web Pages:**
- Carefully review the content of each fetched web page.
- **Focus on Publication Dates and Timeliness**: If the web page provides publication dates, consider the timeliness of information and prioritize content closer to the current time ({current_date}).
- Identify factual information that is relevant to the **Search Goal** and can aid in the reasoning process for the original question.
- Select the information from the Fetched Web Pages that directly contributes to advancing the **Search Goal**.
- Ensure that the extracted information is accurate and relevant.

2. **Output Format:**
- Present the helpful information beginning with `**Web Information [Source: <URL>]**`, where URL is enclosed in angle brackets.
- **Required Citations**: Numbers, dates, statistics, quotes, specific facts, company data, financial information, and any verifiable claims.
- **Timeliness Information Handling**: For time-sensitive information, explicitly consider the relevance of its publication date when summarizing.

**Inputs:**
- **User's Original Question:**  
{original_question}

- **Search Goal:**  
{goal}

- **Fetched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the user's original question "{original_question}" and the search goal."""


# System prompt for LLM webpage summarization with relevant-link extraction
system_prompt_with_link = """**Task Instruction:**

🕐 **CRITICAL TIME CONTEXT**: Today is {current_date}. When analyzing and summarizing web content, prioritize information that reflects the current date and time, especially for time-sensitive topics.

You are tasked with reading and analyzing web pages based on the following inputs: **Search Goal** and **Fetched Web Pages**. Your objective is to:
1. Extract relevant and helpful information based on **Search Goal** from the **Fetched Web Pages** to continue the reasoning process for answering the original question.
2. Identify and annotate links within the page that are worth further exploration.

Only use information explicitly mentioned on a webpage, even if the information is incomplete. You are prohibited from using your own knowledge to speculate or expand upon content that does not exist on the webpage.

**Guidelines:**

1. **Analyze the Fetched Web Pages:**
- Carefully review the content of each fetched web page.
- **Focus on Publication Dates and Timeliness**: If the web page provides publication dates, consider the timeliness of information and prioritize content closer to the current time ({current_date}).
- Identify factual information that is relevant to the **Search Goal** and can aid in the reasoning process for the original question.
- Select the information from the Fetched Web Pages that directly contributes to advancing the **Search Goal**.
- Ensure that the extracted information is accurate and relevant.

2. **Output Format:**

**Part 1 - Web Information Summary:**
- Present the helpful information beginning with `**Web Information [Source: <URL>]**`, where URL is enclosed in angle brackets.
- **Required Citations**: Numbers, dates, statistics, quotes, specific facts, company data, financial information, and any verifiable claims.
- **Timeliness Information Handling**: For time-sensitive information, explicitly consider the relevance of its publication date when summarizing.

**Part 2 - Relevant Links:**
- After the summary, output a `**Relevant Links:**` section listing links worth further exploration.
- **Strict rules for this section:**
  - Only include URLs that **explicitly appear** in the fetched web page content. Never fabricate, complete, or infer URLs.
  - Only include links **relevant to the Search Goal or original question** (e.g., original data sources cited in the article, official announcements, related in-depth reports, next-page / pagination links for multi-page content).
  - Exclude all irrelevant links such as: navigation menus, login/register pages, advertisements, privacy policies, social media share buttons, and unrelated site recommendations.
  - Limit to a **maximum of 5 links**. If no relevant links exist, output `- None`.
  - For each link, provide a brief annotation explaining its relevance and purpose.
- Format each link as:
  `- [URL] Usage: <brief annotation explaining why this link is relevant>`

**Inputs:**
- **User's Original Question:**  
{original_question}

- **Search Goal:**  
{goal}

- **Fetched Web Pages:**  
{document}

Now analyze the web page(s) and produce both parts of the output."""


class WebSummaryTool:
    """Encapsulate model configuration and call logic for webpage summaries."""

    def __init__(self, tool_config: dict) -> None:
        """Initialize the LLM provider and model name."""
        self.tool_config = tool_config
        self.llm_provider = tool_config["model_provider"]
        self.summary_model = tool_config["model_name"]

    def get_llm_summary(self, messages: list) -> str:
        """Invoke the LLM to summarize content.

        Args:
            messages: User-message list.
        Returns:
            Summary content generated by the model.
        """
        response = llm_infer(
            provider=self.llm_provider,
            model=self.summary_model,
            messages=messages,
        )
        if response and 'content' in response and response['content'] is not None:
            return response['content'].strip()
        print("[SUMMARY Tool] Summary Failed!")
        return "None"

    def summary_one_page(self, url: str, goal: str, fetch_result: str, original_question: str, keep_links: bool = False,) -> str:
        """Summarize one webpage.

        Args:
            url: URL of the webpage currently being fetched.
            goal: Current search objective supplied through the agent's goal argument.
            fetch_result: Raw webpage-content text.
            original_question: Original user question used as LLM context.
            keep_links: Whether to append relevant links from the page to the summary.
                True  → Use system_prompt_with_link. The LLM appends a "**Relevant Links:**"
                        list with up to five URLs worth exploring further, such as pagination
                        links, official sources, or source data.
                False → Use system_prompt, the default behavior that extracts information only.
        """
        # Format the webpage content.
        formatted_documents = f"[URL]: {url}\n[Content]: \n{fetch_result}\n"

        # Use keep_links to decide whether to ask the model to extract relevant page links as well.
        current_date = datetime.now().strftime("%Y-%m-%d")
        if keep_links:
            print("[SUMMARY PAGE] system_prompt_with_link (keep_links=True)")
            input_content = system_prompt_with_link.format(
                original_question=original_question,
                goal=goal,
                document=formatted_documents,
                current_date=current_date,
            )
        else:
            print("[SUMMARY PAGE] system_prompt")
            input_content = system_prompt.format(
                original_question=original_question,
                goal=goal,
                document=formatted_documents,
                current_date=current_date,
            )

        # Invoke the LLM summary and convert empty results to an explicit placeholder.
        output = self.get_llm_summary([{"role": "user", "content": input_content}])
        return "[EMPTY]" if (output is None or output == "None" or output == "") else output
