---
title: "Fine-tuning LLM vs In-context learning"
date: 2023-07-22T20:40:19+05:45
ShowToc: true

author: ["CohleM"]
---





This is my experience and experimentation that I did while building a product for the use case of using LLMs for our own data for question answering. If you're doing something similar, this could be of some help.

The most commonly used methods while using LLMs with our own data were typically

*  **Fine-tuning the model with your own data**
*  **Using Retrieval Augmented Generation (RAG) techniques**

## Fine-tuning the model with your own data

This is the initial method and follows the general structure of training a model

* Data preparation 
* Train 
* Evaluate

For my task, I chose to train **OpenAI's davinci base model**

There were mostly no hyperparameters to tune, as OpenAI takes care of it outside the box. Training the model involved more instruction tuning, instructing the model to act in a similar way, rather than just training the model to save data in its model weights. The effectiveness of instruction tuning mostly depended on the data preparation process. Data preparation involved formatting the data into pairs of *<instruction, completion>*.

This is a crucial step for better outputs and depends on the size of the training data. When the model was trained with a small amount of data, it mostly followed the instructions but had limited knowledge and memory of facts from the dataset. In most cases, while testing the model, it followed the instructions but often produced incorrect answers. When the scale of data was increased, it started to follow both the instructions and retain more knowledge. So, this process was crucial in identifying when training OpenAI's models worked best. If you're using it to train on a small scale of data, this would not yield the desired output, and I would recommend using the other process I employed for small-sized data.

## Using Retrieval Augmented Generation (RAG) techniques 
This technique is sometimes referred to as In-context Learning.
This is one of the most trending topics since the release of ChatGPT, and you might have heard the phrase "chat with your own data." This process is simple yet very effective when you have your own small-scale data, which I used for question answering. The process involves:

### Dividing the data into chunks
Here, the format of the data doesn't really matter. The only thing we need to take care of is the size of the chunks. The chunk size should always be smaller than the context length of LLMs, providing space for prompt and completion texts. In my case, the context length of gpt-3.5-turbo was 4,096 tokens, and I divided the chunks into token sizes of 1000. I chose a token size of 1000 for and retrieved top-3 chunks to be passed to the LLM.

### Converting the chunks into embeddings
This process generates embeddings, which are vector representations of our chunks. I experimented with a couple of embedding models, each having its pros and cons. My recommendations for each of the embedding models are as follows:

1. **all-MiniLM-L12-v2**: Useful when you need fast conversion from chunks to embeddings. It has a relatively small dimension of 384 and does a decent job in converting to embeddings.

2. **OpenAI's text-embedding-ada-002**: Useful when you need to generate highly accurate embeddings. If you are using it in real-time, it would be too slow due to its high dimension size of 1536, and API calling makes it even slower.

3. **Instructor**: Useful when you need the accuracy level of text-embedding-ada-002 and fast conversion from text to embeddings. This model is blazingly fast and would save on cost when you embed lots of data.

I went with the Instructor-XL model.

### Storing the embeddings
Many vector database companies have risen around this use case of storing embeddings, such as Pinecone, Chroma, and many more. The trend is to follow the hype and opt for vector databases, which, in fact, are completely useless. If your embeddings are really big, I would recommend using vector databases; otherwise, for medium to small-scale data, ndarray would do a great job. I decided to just use a numpy array.

### Retrieval
Relevant chunks are retrieved based on the similarity between the user's query's embeddings and the chunks' embeddings. Metrics like dot product, cosine similarity, are widely adopted, whereas cosine similarity would suffice. After evaluating the similarity, the top-k chunks are retrieved. We can use reranking to improve the quality of relevant chunks, but top-k would suffice. After retrieving, we are ready to feed the retrieved chunks and the user's query to LLM by combining them with prompting.

### Prompting
Prompting has been the most important step in getting the desired output. For me, this has turned out to be harder than doing research, writing code, and debugging combined. Writing quality prompts is hard. I experimented with a couple of prompting techniques in this project.

Few-shot prompting, few-shot combined with Chain of Thought (CoT) prompting, but I couldn't achieve the desired output. I noticed some problems with these techniques for my use case. These prompting techniques caused too many logic jumps, and the desired logic was never analyzed by the LLM. What worked for me was the map-reduce method and double prompting (which involves calling LLM twice, with 2 prompts, where the latter prompt is combined with the former LLM output). Both methods worked fine, with map-reduce being more expensive. I opted for double prompting and was able to generate the desired output. So, prompting has been challenging for me. Just a thought: maybe we should do some reverse engineering someday and train the prompts with the desired output as context, as described in the [AUTOPROMPT](https://arxiv.org/abs/2010.15980) paper 😅.
