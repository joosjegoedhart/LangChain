# LangChain Tutorials and Experiments

Repository for experimenting with LangChain. I use `Python 3.9`

## Models
LangChain differentiates between 3 types of *models*: LLMs, Chat models and Text embedding models. 

## Prompts
LLM inputs are managed using prompts. `PromptTemplates` help construct prompts from multiple components and can be zero-shot or few-shot.  

## Chains 
Chains combine LLMs with other components for application creation. For example
- Combining LLMs with prompt templates
- Combining multiple LLMs sequentially by taking the first LLM’s output as the input for the second LLM
- Combining LLMs with external data, e.g., for question answering
- Combining LLMs with long-term memory, e.g., for chat history. BY keeping all/ k conversations or by summarizing. 

## Agents
LLMs gebreken can (partly) be omzeild by using supplementary tools such as search, calculators and lookup. Agents decide when to use which tool.  

## Tutorial Links

- [Towards Data Science](https://towardsdatascience.com/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c#bd03)
