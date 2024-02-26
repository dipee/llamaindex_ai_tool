from dotenv import load_dotenv, find_dotenv, get_key
import os
import pandas as pd 
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str 
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms import openai
from prompts import context
os.environ["OPENAI_API_KEY"] = get_key(find_dotenv(), "OPENAI_API_KEY")



population_path = os.path.join('data', 'population.csv')

population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)

population_query_engine.update_prompts({"pandas_prompt": new_prompt})


population_query_engine.query("What is the population of canada")
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This gives the imformation about the world opulation and demographics",
        )
    )
]

llm = openai.OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(llm=llm, tools=tools, verbose=True, context=context)

while (prompt := input("Enter a prompt q to (quit): ")) != "q":
    result = agent.query(prompt)
    print(result)