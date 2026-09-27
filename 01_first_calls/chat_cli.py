"""An interactive multi-turn chatbot. Shows that YOU hold the conversation state, not the API."""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fwlearn import MODEL, client

fw = client()
messages = [{"role": "system", "content": "You are a friendly tutor who teaches LLM inference concepts."}]

print(f"Chatting with {MODEL}. Type 'quit' to exit, 'reset' to clear history.\n")
while True:
    user = input("you> ").strip()
    if user.lower() in {"quit", "exit"}:
        break
    if user.lower() == "reset":
        messages = messages[:1]
        print("(history cleared)\n")
        continue
    messages.append({"role": "user", "content": user})

    print("bot> ", end="", flush=True)
    reply = []
    for chunk in fw.chat.completions.create(model=MODEL, messages=messages, stream=True,
                                            reasoning_effort="none"):
        if chunk.choices and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            reply.append(piece)
            print(piece, end="", flush=True)
    print("\n")

    # Append the assistant turn so the model "remembers" it next time.
    messages.append({"role": "assistant", "content": "".join(reply)})
