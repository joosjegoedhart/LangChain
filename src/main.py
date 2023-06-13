import os
from langchain import HuggingFaceHub, PromptTemplate, ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.agents import load_tools, initialize_agent, AgentType

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Models: LLMs
llm = HuggingFaceHub(repo_id = "google/flan-t5-base")
prompt = "What am I never gonna do?"
completion = llm(prompt)
print(completion)

# Models: Embedding
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
text = "Alice has a parrot"
text_embedding = embeddings.embed_query(text)
print(len(text_embedding))

# Prompts: zero-shot
template = "What is a good name for a company that makes {product}?"

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)
output = prompt.format(product="socks")
print(prompt.format(product = "socks"))

# Chains: LLMs with prompt templates
chain = LLMChain(llm = llm,
                 prompt = prompt)
output = chain.run("colorful things")
print(output)

# Chains: SimpleSequentialChain
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

catchprase = overall_chain.run("colorful things")

# Chains: indexing, i.e. external data
# Create Faiss vector database for documents using Huggingface model for embeddings
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
documents = loader.load()
db = FAISS.from_documents(documents, embeddings)

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True
)

query = 'What am I never going to do?'
result = qa({"query": query})
print(result['result'])

# Chains: long-term meory (e.g. chat history)
conversation = ConversationChain(llm=llm, verbose=True)
conversation.predict(input="Alice has a parrot")
conversation.predict(input="Bob")
answer = conversation.predict(input="What pet does Alice have?")
print(answer)

# Agents: this will give an error for google-flan-t5 model
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)
agent.run("When was Barack Obama born?")