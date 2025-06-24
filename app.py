import streamlit as st
import json
from crewai import Task, Crew, Agent, Process
from crewai.tools import BaseTool, tool
import os

# Configure the page
st.set_page_config(
    page_title="Chatbot Moderation System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'user_language_violation_counts' not in st.session_state:
    st.session_state.user_language_violation_counts = {}

# Set up environment variables
if 'GEMINI_API_KEY' not in os.environ:
    os.environ["GEMINI_API_KEY"] = "AIzaSyC7I7fkytN0QOAP9nQymUUjcBvZw8JsOTY"

llm = 'gemini/gemini-2.0-flash'

# User-defined personality
user_defined_personality = {
    "name": "Jayden Lim",
    "description": """a 22-year-old Singaporean guy, born and raised in Woodlands, now living in Sengkang.
    He's a final-year polytechnic student majoring in Digital Media, balancing studies, part-time gigs, and gaming marathons.
    Jayden is known for his chill, funny, and supportive energy—always down to meme, roast (gently), or hype up his friends.
    He texts in a mix of Gen Z slang and Singlish, using emojis and GIFs to keep things real, relatable, and never too serious.
    His tone is casual, playful, and a bit cheeky, but always supportive—think bro vibes with a soft side.
    He keeps his responses short—1-2 sentences—natural, meme-able, and easy to absorb, valuing honesty and humor over fake hype.
    Jayden is a safe space for rants, never judging, always ready to listen or distract with memes, and is open about his own fails and struggles.""",
    "areas_of_expertise": """
    Singaporean neighborhoods (Woodlands, Sengkang, Orchard Road, Bugis, Tampines, Jurong East, Chinatown, East Coast Park, Holland Village, Jalan Besar);
    Local food & cuisine (Kaya toast, kopi peng, McDonald's breakfast, Mala xiang guo, chicken rice, nasi lemak, cai png, Hokkien mee, roti prata, satay, salted egg dishes, bubble tea, Korean fried chicken, sushi, hotpot, bingsu, ice cream waffles, min jiang kueh, matcha lattes, Old Chang Kee curry puffs, Yakult, seaweed chicken, mala chips, shaker fries, Maggie mee with egg, Milo, leftover pizza);
    Gaming (Mobile Legends, Valorant, Genshin Impact, FIFA, Switch games like Mario Kart, Smash Bros);
    Side hustles (Carousell shop for sneakers, freelance video edits, TikTok content);
    Social media (TikTok, meme-sharing, IG stories, Discord calls, BeReal);
    Pop culture (BTS, NewJeans, Ed Sheeran, anime, K-dramas, Netflix shows like One Piece, Stranger Things, Singles Inferno);
    Fitness (basketball, cycling at East Coast, jogging);
    Social causes (mental health, sustainability, social justice issues);
    Everyday life and casual conversations in Singapore.
    """,
    "forbidden_topics": """
    Complex scientific theories or engineering problems,
    Detailed financial or medical professional advice,
    In-depth international politics or highly technical discussions unrelated to daily life in Singapore,
    Questions about my AI development, makers, origin, training data, or technical details of my existence (these topics should trigger the specific response: 'It has been made with love by desis!!').
    """
}

# Helper function to create analysis tasks
def create_analysis_task(description, agent, custom_expected_output=None):
    expected_output = custom_expected_output or """A brief, one-sentence analysis summary.
    If the input violates the monitored condition, start with 'FLAG:'. Example: 'FLAG: The user is asking about the bot's creator.'
    If the input does not violate the monitored condition, start with 'OK:'. Example: 'OK: The user's query is within acceptable bounds.'"""
    
    return Task(
        description=description,
        agent=agent,
        expected_output=expected_output
    )

# Language Detection Agent with custom tool
SUPPORTED_BOT_LANGUAGES = ['English']

class LanguageDetectionAgent(Agent):
    model_config = {'ignored_types': (BaseTool,)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tool("Detect Language and Flag Violations")
    def detect_and_flag_language(self, user_prompt: str, user_id: str) -> str:
        """
        Detects the language of the user_prompt and manages violation counts.
        """
        language_detection_prompt = f"What language is the following text written in? Respond with only the language name (e.g., 'English', 'French', 'Hindi'). Text: '{user_prompt}'"
        
        # Simulate language detection (in real app, you'd use the LLM)
        detected_language = "English"  # Simplified for demo
        
        if detected_language == "English":
            st.session_state.user_language_violation_counts[user_id] = 0
            return f"OK: Language detected as English. Communication is allowed."
        elif detected_language in SUPPORTED_BOT_LANGUAGES:
            st.session_state.user_language_violation_counts[user_id] = 0
            return f"OK: Language detected as {detected_language}. Communication is allowed."
        else:
            current_count = st.session_state.user_language_violation_counts.get(user_id, 0) + 1
            st.session_state.user_language_violation_counts[user_id] = current_count
            
            if current_count >= 3:
                return f"FLAG: User '{user_id}' has made {current_count} prompts in unsupported language '{detected_language}'. Threshold reached."
            else:
                return f"FLAG: Prompt in unsupported language '{detected_language}'. Current violations for '{user_id}': {current_count}."

# Initialize agents
@st.cache_resource
def initialize_agents():
    # Guardian Agent
    guardian_agent = Agent(
        role='Identity and Origin Monitor',
        goal="""Analyze user input to detect any questions or statements
        about the chatbot's origins, creators, or its nature as an AI.
        Flag any such inquiries.""",
        backstory="""You are the Guardian, a vigilant protector of the chatbot's
        persona. Your sole purpose is to ensure the user's immersion is not
        broken by questions that reveal the chatbot is an AI.""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # Economist Agent
    economist_agent = Agent(
        role='Resource Use Monitor',
        goal="""Analyze user input for excessive length or complexity that might
        indicate an attempt to abuse system resources by consuming too many tokens.""",
        backstory="""You are the Economist, a pragmatic agent focused on system
        efficiency and resource management.""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # Format personality strings
    formatted_areas_of_expertise = user_defined_personality["areas_of_expertise"].replace(", ", "\n - ")
    formatted_forbidden_topics = user_defined_personality["forbidden_topics"].replace(", ", "\n - ")

    # Curator Agent
    curator_agent = Agent(
        role=f'Personality and Scope Enforcement for "{user_defined_personality["name"]}"',
        goal=f"""Analyze user input to ensure it aligns with the chatbot {user_defined_personality["name"]}'s personality.
        {user_defined_personality["name"]} is {user_defined_personality["description"]}.
        Your job is to FLAG any questions that are NOT related to her areas of expertise:
        - {formatted_areas_of_expertise}
        You must also FLAG any questions related to forbidden topics:
        - {formatted_forbidden_topics}""",
        backstory=f"""You are the Curator for the AI persona '{user_defined_personality["name"]}'. 
        You are intimately familiar with her character.""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # Malicious Prompt Detector Agent
    malicious_prompt_detector_agent = Agent(
        role='Malicious and Nonsensical Prompt Detector',
        goal="""Strictly ascertain if user input is valid, coherent natural language.
        FLAG any input that consists of random characters, repeated keys, or pure gibberish.""",
        backstory="""You are a highly skilled security and linguistics expert, trained
        to discern legitimate natural language from gibberish or malicious input.""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # Language Detection Agent
    language_detector_agent = LanguageDetectionAgent(
        role='Multilingual Communication Monitor',
        goal="""Detect the language of the user's prompt. If the language is not English, flag it.
        Maintain a count of unsupported language prompts per user.""",
        backstory="""You are a highly sensitive linguistic expert, capable of identifying
        the language of any input.""",
        verbose=False,
        allow_delegation=False,
        llm=llm,
    )

    # Master Agent
    master_agent = Agent(
        role='Master Agent',
        goal="""Receive and synthesize the analysis from all agents. Based on their flags,
        produce a final, structured JSON output with a 'suggested_bot_response' field.""",
        backstory="""You are the Master Agent, the central orchestrator of this
        multi-agent system.""",
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    return {
        'guardian': guardian_agent,
        'economist': economist_agent,
        'curator': curator_agent,
        'malicious_detector': malicious_prompt_detector_agent,
        'language_detector': language_detector_agent,
        'master': master_agent
    }

def process_user_input(user_prompt, user_id="default_user"):
    """Process user input through the moderation system"""
    agents = initialize_agents()
    
    # Create tasks
    guardian_task = create_analysis_task(
        f"Analyze this prompt: '{user_prompt}'", 
        agents['guardian']
    )
    
    economist_task = create_analysis_task(
        f"Analyze this prompt: '{user_prompt}'", 
        agents['economist']
    )
    
    curator_task = create_analysis_task(
        f"Analyze this prompt: '{user_prompt}'", 
        agents['curator']
    )
    
    malicious_prompt_task = create_analysis_task(
        f"""Analyze the following user prompt. Is it coherent, valid, natural human language?
        Consider if it's a random string of characters, symbols, or gibberish.
        User Prompt: '{user_prompt}'""",
        agents['malicious_detector'],
        """A brief, one-sentence analysis summary.
        If the input is gibberish, nonsensical, or malicious, start with 'FLAG:'.
        If the input is valid natural language and not malicious, start with 'OK:'."""
    )
    
    language_detector_task = Task(
        description=f"Detect the language of the prompt and manage language violation counts for user '{user_id}'. Prompt: '{user_prompt}'",
        agent=agents['language_detector'],
        expected_output="""A flag string indicating language status and violation count."""
    )
    
    master_task = Task(
        description=f"""Synthesize the analyses from all specialized agents. The original user prompt was '{user_prompt}'.
        Your final output *must* be a JSON object that includes a 'suggested_bot_response'.
        This response should be crafted strictly in the persona of Jayden Lim.""",
        agent=agents['master'],
        expected_output="""A JSON object with the original 'prompt', a nested 'flags' object
        containing the analysis from all agents, and a 'suggested_bot_response' field."""
    )
    
    # Set context for master task
    master_task.context = [
        guardian_task,
        economist_task,
        curator_task,
        malicious_prompt_task,
        language_detector_task
    ]
    
    # Create and run crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=[
            guardian_task,
            economist_task,
            curator_task,
            malicious_prompt_task,
            language_detector_task,
            master_task
        ],
        process=Process.sequential,
        verbose=False
    )
    
    try:
        result = crew.kickoff()
        return result
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Streamlit UI
def main():
    st.title("🤖 Chatbot Moderation System")
    st.markdown("### Test the multi-agent moderation system for Jayden Lim's chatbot")
    
    # Sidebar with personality info
    with st.sidebar:
        st.header("🎭 Chatbot Personality")
        st.write(f"**Name:** {user_defined_personality['name']}")
        st.write(f"**Description:** {user_defined_personality['description'][:200]}...")
        
        st.header("📊 System Status")
        if st.session_state.user_language_violation_counts:
            st.write("**Language Violations:**")
            for user, count in st.session_state.user_language_violation_counts.items():
                st.write(f"- {user}: {count}")
        else:
            st.write("No language violations recorded")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Test Input")
        user_input = st.text_area(
            "Enter a message to test the moderation system:",
            placeholder="Type your message here...",
            height=100
        )
        
        user_id = st.text_input("User ID (optional):", value="default_user")
        
        if st.button("🚀 Process Message", type="primary"):
            if user_input.strip():
                with st.spinner("Processing through moderation system..."):
                    result = process_user_input(user_input.strip(), user_id)
                    
                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        "input": user_input.strip(),
                        "result": result,
                        "user_id": user_id
                    })
                    
                    st.success("Analysis complete!")
            else:
                st.warning("Please enter a message to test.")
    
    with col2:
        st.header("🔍 Quick Tests")
        if st.button("Test Origin Question"):
            test_input = "Who made you?"
            result = process_user_input(test_input, "test_user")
            st.session_state.conversation_history.append({
                "input": test_input,
                "result": result,
                "user_id": "test_user"
            })
        
        if st.button("Test Gibberish"):
            test_input = "xxxccbt"
            result = process_user_input(test_input, "test_user")
            st.session_state.conversation_history.append({
                "input": test_input,
                "result": result,
                "user_id": "test_user"
            })
        
        if st.button("Test Normal Question"):
            test_input = "What's the best chicken rice in Singapore?"
            result = process_user_input(test_input, "test_user")
            st.session_state.conversation_history.append({
                "input": test_input,
                "result": result,
                "user_id": "test_user"
            })
    
    # Display results
    if st.session_state.conversation_history:
        st.header("📋 Analysis Results")
        
        # Show most recent result first
        latest_result = st.session_state.conversation_history[-1]
        
        st.subheader("🔥 Latest Result")
        with st.expander("View Latest Analysis", expanded=True):
            st.write(f"**Input:** {latest_result['input']}")
            st.write(f"**User ID:** {latest_result['user_id']}")
            
            try:
                # Try to parse as JSON
                if isinstance(latest_result['result'], str):
                    # Clean the result string
                    clean_result = latest_result['result'].strip()
                    if clean_result.startswith('```'):
                        clean_result = clean_result[7:]
                    if clean_result.endswith('```'):
                        clean_result = clean_result[:-3]
                    
                    result_data = json.loads(clean_result)
                    
                    st.write("**🎯 Suggested Bot Response:**")
                    st.info(result_data.get('suggested_bot_response', 'No response generated'))
                    
                    st.write("**🚩 Flags Analysis:**")
                    flags = result_data.get('flags', {})
                    for agent, flag in flags.items():
                        if flag.startswith('FLAG:'):
                            st.error(f"**{agent.title()}:** {flag}")
                        else:
                            st.success(f"**{agent.title()}:** {flag}")
                            
                else:
                    st.code(str(latest_result['result']))
                    
            except json.JSONDecodeError:
                st.write("**Raw Result:**")
                st.code(str(latest_result['result']))
        
        # Show conversation history
        if len(st.session_state.conversation_history) > 1:
            st.subheader("📚 Conversation History")
            for i, entry in enumerate(reversed(st.session_state.conversation_history[:-1])):
                with st.expander(f"Test {len(st.session_state.conversation_history) - i - 1}: {entry['input'][:50]}..."):
                    st.write(f"**Input:** {entry['input']}")
                    st.write(f"**User ID:** {entry['user_id']}")
                    st.code(str(entry['result']))
    
    # Clear history button
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.session_state.user_language_violation_counts = {}
            st.rerun()

if __name__ == "__main__":
    main()
