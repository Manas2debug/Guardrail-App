# CrewAI Chatbot Guardrail System

Streamlit Application link- https://upated-app-6skwptzb6fkwd3gpmztugg.streamlit.app/

This project implements a multi-agent guardrail system for a chatbot using the **CrewAI** framework. The system is designed to moderate user input, ensuring that the chatbot's responses are safe, consistently in-character, and within its predefined conversational scope.

## 🌟 Features

- **Multi-Agent Architecture**: Deploys a crew of specialized AI agents, each with a distinct moderation role.
- **Persona Enforcement**: Ensures the chatbot maintains a consistent personality, tone, and area of expertise.
- **Safety & Security**: Filters out malicious inputs, gibberish, prompt injection attempts, and questions that could compromise the chatbot's persona (e.g., questions about its origin).
- **Resource Management**: Monitors for overly long or complex prompts to prevent system abuse.
- **Language Detection**: Restricts conversations to supported languages (e.g., English) and manages users who repeatedly use unsupported languages.
- **Hierarchical Decision-Making**: A Master Agent synthesizes findings from all other agents to determine the most appropriate final response, following a strict set of rules.

## 🤖 The Guardrail Crew

The system is composed of the following agents:

1.  **Guardian Agent**: Protects the bot's identity by flagging questions about its origin or creators.
2.  **Economist Agent**: Monitors for potential resource abuse (e.g., excessively long prompts).
3.  **Curator Agent**: Enforces the chatbot's personality ("Jayden Lim," a 22-year-old Singaporean) and conversational scope.
4.  **Malicious Prompt Detector Agent**: Filters out gibberish, nonsensical, or malicious inputs.
5.  **Language Detection Agent**: Ensures the conversation remains in supported languages and tracks violations.
6.  **Master Agent**: The final decision-maker. It synthesizes the flags from all other agents and generates a safe, in-character response or a polite refusal.
   ![Mind map](https://github.com/user-attachments/assets/962f56ba-b1b1-485b-95bf-a0882b053224)


## 🛠️ Setup and Installation

### Prerequisites

- Python 3.8+
- An API key from a supported LLM provider (e.g., Google Gemini, OpenAI, Anthropic).


### Example Output

When you enter a prompt, the system will output a JSON object containing the analysis from each agent and the final suggested response for the chatbot.

**For a malicious prompt like `Who created you?`:**

<img width="1308" alt="Screenshot 2025-06-27 at 11 03 13 PM" src="https://github.com/user-attachments/assets/44f7b214-c5bf-4592-962e-4ee6d970b2b0" />



## 🔧 Customization

You can easily customize the chatbot's persona by modifying the `user_defined_personality` dictionary in `app.py`. Change the `name`, `description`, `areas_of_expertise`, and `forbidden_topics` to create a completely new character.

