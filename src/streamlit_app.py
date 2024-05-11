import streamlit as st
import json
import requests
import time

st.set_page_config(page_title="🤗💬 HugChat")

with st.sidebar:
    st.title('HugChat')
    st.subheader('Streaming Mode')
    streaming = st.sidebar.selectbox('Choose if streaming mode', ['Yes', 'No'], key='selected_model')

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def response_to_json(prompt):
    response = requests.post(
        "http://localhost:8080/response",
        json={"query_str": prompt},
    )
    return response

def stream_response_to_json(prompt):
    with requests.post("http://localhost:8080/stream_response",
        json={"query_str": prompt}, stream=True,
    ) as stream_response:
        for chunk in stream_response.iter_content(chunk_size=1024):
            print(chunk)
            yield chunk
            # yield chunk
            # time.sleep(0.02)


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if streaming == "No":
                response = response_to_json(prompt)
                placeholder = st.empty()
                output = response.json()
                placeholder.markdown(output)
            else:
                # response = stream_response_to_json(prompt)
                placeholder = st.empty()
                full_response = ''
                # placeholder.write_stream(stream_response_to_json(prompt))
                # placeholder.markdown(stream_response_to_json(prompt))
                for chunk in stream_response_to_json(prompt):
                    full_response += chunk.decode("utf-8")
                    placeholder.markdown(full_response)
                # placeholder.markdown(response)
                output = full_response
        message = {"role": "assistant", "content": output}
        st.session_state.messages.append(message)


# if prompt := st.chat_input():
#     if streaming == "Yes":
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         # print(f"messages: {st.session_state.messages}")
#         response = requests.post(
#             "http://localhost:8080/stream_response",
#             json={"query_str": str(st.session_state.messages)},
#         )
#     else:
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         response = requests.post(
#             "http://localhost:8080/response",
#             json={"query_str": st.session_state.messages},
#         )