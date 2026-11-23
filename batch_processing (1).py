#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
from openai import OpenAI


# In[2]:


import os

API_KEY = os.getenv("OPENAI_API_KEY")


# In[23]:


# Load CSV and subset
df = pd.read_csv("Y12-15_intake24_grouped_output.csv")
df = df[['ID', 'RecallNumber', 'MealName', 'SearchTerm', 'DescriptionEn','potential_mismatch']]


# In[26]:


# Initialize OpenAI client

client = OpenAI(api_key=API_KEY)


# Function to format a batch prompt
def format_batch_prompt(batch_df):
    examples = []
    for i, row in enumerate(batch_df.itertuples(index=False)):
        example = f"{i+1}.\nSearchTerm: {row.SearchTerm}\nRecorded: {row.DescriptionEn}"
        examples.append(example)
    return "\n\n".join(examples)

# # Parse model output back into list
# def parse_model_output(output, expected_count):
#     lines = output.strip().split("\n")
#     results = [""] * expected_count
#     for line in lines:
#         if not line.strip():
#             continue
#         if line[0].isdigit():
#             try:
#                 index, content = line.split(".", 1)
#                 idx = int(index.strip()) - 1
#                 if 0 <= idx < expected_count:
#                     results[idx] = content.strip()
#             except:
#                 continue
#     return results

def parse_model_output(output, expected_count):
    lines = output.strip().split("\n")
    results = [{"value": "", "reason": ""} for _ in range(expected_count)]
    for line in lines:
        if not line.strip():
            continue
        if line[0].isdigit():
            try:
                index, content = line.split(".", 1)
                idx = int(index.strip()) - 1
                if 0 <= idx < expected_count:
                    # Split on the first comma
                    parts = content.strip().split(",", 1)
                    value = parts[0].strip()
                    reason = parts[1].strip() if len(parts) > 1 else ""
                    results[idx] = {"value": value, "reason": reason}
            except:
                continue
    return results

# Create a column for results
df['LostFoodsAnalysis'] = ""

# Process in batches of 10
batch_size = 10
total_size = 300
#total_size = len(df)


# # Prompt template
# SYSTEM_PROMPT = (
#     "You are an assistant that checks for missing food items in records of food made by study participants searching for the foods they have eaten.\n"
#     "For each example, compare the foods searched for (SearchTerm) to the recorded foods (Recorded).\n"
#     "List any foods from the search that are missing from the recorded list. If nothing is missing, leave the output blank.\n"
#     "Note that the search terms might have spelling mistakes. A search for 'watee' likely matches a recorded item of 'water'.\n"
#     "As an LLM you will also understand semantic matches: A search for 'chicken curry' would match 'chicken madras'.\n"
#     "You can also understand approximate matches like 'cheese' and 'Cheshire cheese'.\n"
#     "If you think you have a found a missing item, check it again carefully against the recorded food to make sure it is actually missing before returning it."
#     "Return your results in the format:\n1. [missing foods or blank]\n2. [missing foods or blank]\n... up to {batch_size}."
# )

# # Prompt template
# SYSTEM_PROMPT = (
#     "You are a food and nutrition expert analyzing dietary recall data. Your task is to determine if there's a mismatch between what a participant searched for and what they actually recorded.\n"
#     "For each example, compare the foods searched for (labelled SearchTerm) to the recorded foods (labelled Recorded).\n"
#     "Consider these as VALID MATCHES (don't flag):\n"
#     "   - Specific versions of general foods (coffee → latte, tea → Earl Grey)\n"
#     "   - Brand names or detailed descriptions (biscuit → digestive biscuit)\n"
#     "   - Common misspellings (warter → water)\n"
#     "   - Broader relationships (curry → chicken balti or cheese → cheddar cheese)\n"
#     "   - Different preparation methods (potato → mashed potato)\n"
#     "Consider these as MISMATCHES (flag):\n"
#     "   - Completely different food items (pizza → ice cream)\n"
#     "   - Missing items from search terms (searched \"bacon, eggs, toast\" but only recorded \"bacon, eggs\")\n"
#     "   - Missing parts from search terms (searched \"chicken fried rice\" but only recorded \"fried rice\")\n"
#     "List any foods from the search list (SearchTerm) that are missing from the recorded list (Recorded). If nothing is missing, leave the output blank.\n"
#     "Do not use commas as a definitive separator for different items, use semantics to determine whole food items.\n"
#     "If you think you have a found a missing item, provide reasoning why you think it is missing and check it again carefully against the recorded food list to make sure it is actually missing before returning it."
#     "Return your results in the format:\n1. [missing foods or blank], [reason or blank]\n2. [missing foods or blank], [reason or blank]\n... up to {batch_size}."
# )

# Prompt template
SYSTEM_PROMPT = (
    "You are a food and nutrition expert analyzing dietary recall data. Your task is to determine if there's a mismatch between what a participant searched for and what they actually recorded.\n"
    "For each example, compare the foods searched for (labelled SearchTerm) to the recorded foods (labelled Recorded).\n"
    "The objective is to stop underreporting of calories for a participant if a searched food is missed.\n"
    "Consider these as VALID MATCHES (don't flag):\n"
    "   - Specific versions of general foods (coffee → latte, tea → Earl Grey)\n"
    "   - Brand names or detailed descriptions (biscuit → digestive biscuit)\n"
    "   - Common misspellings (warter → water)\n"
    "   - Broader relationships (curry → chicken balti or cheese → cheddar cheese)\n"
    "   - Different preparation methods (potato → mashed potato)\n"
    "Consider these as MISMATCHES (flag):\n"
    "   - Completely different food items (pizza → ice cream)\n"
    "   - Missing items from search terms (searched \"bacon, eggs, toast\" but only recorded \"bacon, eggs\")\n"
    "   - Missing parts from search terms (searched \"chicken fried rice\" but only recorded \"fried rice\")\n"
    "List any foods from the search list (SearchTerm) that are missing from the recorded list (Recorded).\n"
    "If you think you have a found a missing food or part of food that would result in reduced calories being reported, provide reasoning why you think it is missing and check it again carefully against the recorded food list to make sure it is actually missing before returning it.\n"
    "If nothing is missing, leave the output blank\n.
    "Return your results in the format:\n1. [missing foods or blank], [reason or blank]\n2. [missing foods or blank], [reason or blank]\n... up to {batch_size}."
)


for start in range(0, total_size, batch_size):
    end = start + batch_size
    batch_df = df.iloc[start:end].copy()
    if batch_df.empty:
        continue
    
    print(f"Processing rows {start+1} to {min(end, total_size)} of {total_size}...")
    
    prompt = format_batch_prompt(batch_df)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        output = response.choices[0].message.content.strip()
        parsed_results = parse_model_output(output, len(batch_df))


    #     # Assign results back to the main DataFrame
    #     df.loc[start:end-1, 'LostFoodsAnalysis'] = parsed_results
        parsed_df = pd.DataFrame(parsed_results)
        df.loc[start:end-1, ['LostFoodsAnalysis', 'LostFoodsReason']] = parsed_df.values
    # except Exception as e:
    #     df.loc[start:end-1, 'LostFoodsAnalysis'] = f"Error: {e}"
    except Exception as e:
        df.loc[start:end-1, 'LostFoodsAnalysis'] = f"Error: {e}"
        df.loc[start:end-1, 'LostFoodsReason'] = f"Error: {e}"

    time.sleep(1)  # avoid rate limits

# Save results
df.to_csv("lost_foods_output_batch.csv", index=False)


# In[8]:


temp_df = df.head(10)
prompt = format_batch_prompt(batch_df)


# In[19]:


output


# In[ ]:




