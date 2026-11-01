import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    r.raise_for_status()

    response = r.json()

    # Return only the generated answer
    return response["response"]

df = joblib.load("embeddings.joblib")

income_query = input("Ask a Question: ")

question_embedding = create_embedding([income_query])[0]

similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [question_embedding]
).flatten()

top_results = 5 #IN no of videos the trained on the at this time we have only 4 video so we trianed on these 4 videos

# Get more results first
max_idx = similarities.argsort()[::-1][0:top_results]

# Select only unique text
selected = []
seen_text = set()

for idx in max_idx:
    text = df.iloc[idx]["text"]

    if text not in seen_text:
        selected.append(idx)
        seen_text.add(text)

    if len(selected) == top_results:
        break

new_df = df.iloc[selected]

# print(new_df[["title", "number", "text"]].to_string(index=False))

prompt = f'''I am teaching web development in my sigma web development course. Here are video subtitle chunks containing video title,video number,start time in second,end time in seconds, the text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
-----------------------------
"{income_query}"
User asks this questions related to this videos chunks, you have to answer in a human way (don't mention the above format its just for you) where ans how 
much content is taught in which video (in which video and at what timestamp) and guide the user to go to the particular video.If user asks unrealted question, tell him that you can only answer questions related to the course
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)

print("\n========== AI ANSWER ==========\n")
print(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"],item["start"],item["end"])