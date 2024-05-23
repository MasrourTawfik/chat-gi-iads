langchain 
=====================

**LangChain** is a framework for developing applications powered by large language models (LLMs).
LangChain simplifies every stage of the LLM application lifecycle:
1.Development: Build your applications using LangChain's open-source building blocks and components. Hit the ground running using third-party integrations and Templates.
2.Productionization: Use LangSmith to inspect, monitor and evaluate your chains, so that you can continuously optimize and deploy with confidence.
3.Deployment: Turn any chain into an API with LangServe.


.. image:: images/100.jpeg
   :width: 50%
   :align: center

Concretely, the framework consists of the following open-source libraries:
1.langchain-core: Base abstractions and LangChain Expression Language.
2.langchain-community: Third party integrations.
3.Partner packages (e.g. langchain-openai, langchain-anthropic, etc.): Some integrations have been further split into their own lightweight packages that only depend on langchain-core.
4.langchain: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
5.langgraph: Build robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.
6.langserve: Deploy LangChain chains as REST APIs.
7.LangSmith: A developer platform that lets you debug, test, evaluate, and monitor LLM applications.
