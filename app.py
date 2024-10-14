import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Sidebar for Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
        st.warning('Please enter a valid API key!', icon='⚠️')
    else:
        st.success('API key provided!', icon='✅')
        os.environ['REPLICATE_API_TOKEN'] = replicate_api

    # Model selection
    selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'])
    llm = 'a16z-infra/llama7b-v2-chat' if selected_model == 'Llama2-7B' else 'a16z-infra/llama13b-v2-chat'
    
    # Parameters
    temperature = st.slider('Temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('Max Length', min_value=32, max_value=128, value=120, step=8)

# Store LLM responses in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Button to clear chat history
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function to generate LLaMA2 response
def generate_llama2_response(prompt_input):
    dialogue_history = "You are a helpful assistant.\n"
    for msg in st.session_state.messages:
        dialogue_history += f"{msg['role'].capitalize()}: {msg['content']}\n"
    output = replicate.run(llm, input={"prompt": f"{dialogue_history}{prompt_input}\nAssistant:", "temperature": temperature, "top_p": top_p, "max_length": max_length})
    return output

# User prompt input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate response if the last message is from the user
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
