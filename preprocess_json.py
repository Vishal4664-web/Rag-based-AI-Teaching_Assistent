# # import requests
# # import os
# # import joblib
# # import json
# # import numpy as np
# # import pandas as pd
# # from sklearn.metrics.pairwise import cosine_similarity

# # def create_embedding(text_list):
# #     r = requests.post("http://localhost:11434/api/embed",
# #         json={
# #             "model": "bge-m3",
# #             "input": text_list
# #         }
# #     )

# #     embedding = r.json()["embeddings"]
# #     return embedding


# # jsons = os.listdir("jsons")
# # print(jsons)
# # my_dicts=[]
# # chunk_id=0

# # for json_file in jsons:

# #     with open(f"jsons/{json_file}") as f:
# #         content = json.load(f)
# #     print(f"Creating Embedding for {json_file}")    

# #     embeddings = create_embedding([c['text'] for c in content])

# #     for i,chunk in enumerate(content):
# #         chunk['chunk_id']=chunk_id
# #         chunk['embedding']=embeddings[i]
# #         chunk_id+=1
# #         my_dicts.append(chunk)
# #         # if(i==5):
# #         #     break
# #     # break # it means only file file will embedding
# # # print(my_dicts)

# # df=pd.DataFrame.from_records(my_dicts)

# # # display_df = df.drop(columns=['embedding'])
# # # print(display_df.to_string(index=False))

# # joblib.dump(df,'embeddings.joblib')

# # # print(df.loc[[197, 75, 258], ["chunk_id", "title", "number", "text"]].to_string())





# import requests
# import os
# import joblib
# import json
# import numpy as np
# import pandas as pd

# def create_embedding(text_list):
#     r = requests.post(
#         "http://localhost:11434/api/embed",
#         json={
#             "model": "bge-m3",
#             "input": text_list
#         }
#     )

#     return r.json()["embeddings"]


# jsons = os.listdir("newjsons")

# my_dicts = []
# chunk_id = 0

# for json_file in jsons:

#     with open(f"newjsons/{json_file}") as f:
#         content = json.load(f)

#     print(f"Creating Embedding for {json_file}")

#     # embeddings = create_embedding([c["text"] for c in content])
#     embeddings = create_embedding([c['text'] for c in content])

#     # print("TEXT 0:", content[0]["text"])
#     # print("EMBEDDING 0:", embeddings[0][:5])

#     # print("TEXT 1:", content[1]["text"])
#     # print("EMBEDDING 1:", embeddings[1][:5])

#     for i, chunk in enumerate(content):
#         chunk["chunk_id"] = chunk_id
#         chunk["embedding"] = embeddings[i]

#         chunk_id += 1
#         my_dicts.append(chunk)

# df = pd.DataFrame.from_records(my_dicts)

# joblib.dump(df, "embeddings.joblib")



import requests
import os
import joblib
import json
import pandas as pd

def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    r.raise_for_status()

    return r.json()["embeddings"]


jsons = os.listdir("newjsons")

my_dicts = []
chunk_id = 0

for json_file in jsons:

    if not json_file.endswith(".json"):
        continue

    file_path = os.path.join("newjsons", json_file)

    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    print(f"Creating Embedding for {json_file}")

    # Extract the actual chunks
    chunks = content["chunks"]

    # Create embeddings from merged chunk text
    embeddings = create_embedding([c["text"] for c in chunks])

    for i, chunk in enumerate(chunks):

        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]

        chunk_id += 1
        my_dicts.append(chunk)

    print(f"Chunks processed: {len(chunks)}")


df = pd.DataFrame.from_records(my_dicts)

joblib.dump(df, "embeddings.joblib")

