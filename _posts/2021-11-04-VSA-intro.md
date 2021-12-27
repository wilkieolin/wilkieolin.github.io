---
layout: post
title: A Brief Introduction to Neuromorphic Computing
categories: computing research
---

The question "what is neuromorphic computing" has a textbook answer: it's research in computing methods which are inspired by the brain. However, the problem with this definition is that because there are so many ways be inspired by the brain, it often appears debatable whether a certain approach is neuromorphic. 

For instance, as their name suggests, the first neural networks were inspired by rate-based interpretations of a biological neuron's firing. And yet, it is rare that these highly-successful networks or their hardware accelerators are referred to as 'neuromorphic.' 

From this outlook, it might appear that what is declared a neuromorphic architecture only reflects on the intent of a researcher to place a work within a certain field or market it to a specific audience. While there may be some truth to that, in my experience there are clear markers of what a constitutes a modern neuromorphic strategy: a focus on utilizing forms of sparsity, achieving distributed computing, and utilizing novel hardware. Below, I explain each of these principles in further detail. 

<!--more-->

# The Motivation for Neuromorphic

Biological computation is highly efficient, both in terms of energy usage and in learning new information; the human brain uses only [tens of watts](https://www.scientificamerican.com/article/thinking-hard-calories/) and is capable of flexibly learning new tasks. This contrasts greatly to modern artificial intelligence (AI) models, which [consume vast amounts of energy to train](https://arxiv.org/abs/1906.02243) and are still faced with the challenge of '[catastrophic forgetting](https://www.sciencedirect.com/science/article/pii/S0893608019300231),' meaning they cannot easily learn new situations or apply their existing knowledge to new tasks.

For example, if a purely hypothetical game 'Starcraft 3' were released tomorrow, I would be confident I could play through its campaign on a normal difficulty level by applying my experience with its predecessor, Starcraft 2. New units or graphics would doubtless be included with the new entry, but I would be able to incorporate these new elements as I played.

In contrast, it is likely that the AI models such as [AlphaStar](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii) currently used to play Starcraft 2 would require moderate or extensive reworking to incorporate these new elements, as well as re-training. 

While this inefficiency might not be a problem for well-defined applications with huge amounts of data available, in other areas where AI could be deployed it's a deal-breaker. [Robotics is one key example](https://spectrum.ieee.org/how-deepmind-is-reinventing-the-robot): a robotic 'agent' deployed in a field needs to be able to reason about new objects and scenarios it hasn't encountered before, and do so safely and reliably. Furthermore, it needs to be able to do this without a cable attaching it to a power plant and/or supercomputer. 

AI faces another challenge: historically, its progress has been [linked to an increasing number of model parameters](https://dl.acm.org/doi/pdf/10.1145/3381831), where each parameter is essentially a 'dial' which needs to be turned to the correct position for the model to produce the right answer. Networks which revolutionized image classification in 2013 used [millions of parameters](https://towardsdatascience.com/understanding-alexnet-a-detailed-walkthrough-20cd68a490aa); in 2021, networks focused on advancing 'natural language processing' (NLP) tasks such as translation use [billions](https://developer.nvidia.com/blog/openai-presents-gpt-3-a-175-billion-parameters-language-model/) or even [trillions](https://arxiv.org/pdf/2101.03961.pdf) of parameters. 

Needing to tune this many parameters places designing and training these models outside the abilities of any but the largest corporations and governments with sufficient resources. Furthermore, the slowdown in transistor scaling which provided faster and cheaper computers for decades makes it likely that these models will remain a challenge to train and deploy even with [specialized hardware](https://ieeexplore.ieee.org/abstract/document/9286149?casa_token=bqJFvwVowKYAAAAA:GvfAWMVDwY1NsdbfyqjzZ7PmYuCx-2AlJ6QWaruu19VOpNKhPt4iInWDXZ8pR3EUrQeDViRV). 

Besides these, other problems exist with modern AI such as [bias](https://hbr.org/2019/10/what-do-we-do-about-the-biases-in-ai), [lack of robustness](https://towardsdatascience.com/breaking-neural-networks-with-adversarial-attacks-f4290a9a45aa), and [lack of explainability](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence). To realize the full potential of AI as reliable, autonomous systems which can improve human quality-of-life in areas such as medicine, manual labor, and household assistance, these issues must be addressed. 

While resolving all these issues may seem insurmountable, we know it must be possible, as currently the human brain is more capable of addressing these issues more effectively than any artificial system. In the current era of AI, neuromorphic computing seeks to these extant issues by applying principles of biological computation. Here, I'll focus on two broad topics which neuromorphic researchers often aim to achieve: sparsity and distribution. 

# Sparsity

Sparsity is a general concept which implies that out of a large set of elements, only a small fraction have values are 'active.' All other elements are inactive, which often means their values are zero or undefined. Either way, computations with these elements depends only on the 'active' elements. 

Having very few active elements is desirable in computation, as this reduces the amount of information which must be computed and transported. The more non-zero values a computation requires, the more energy must be spent communicating these values to the downstream process which require that information. Often, the expense of moving information throughout computing systems is greater than the energy required for other operations - [particularly in AI](https://ieeexplore.ieee.org/abstract/document/8335698?casa_token=OTb7GK9KyNEAAAAA:5pp1G855ulinLu7WiiFtxWzT2qyXHZQzVgR7dbjb3VKI7NF5ARwJ-3N1gpRtd65T73_2fMvb). 

[Sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix) are a common example of sparse computation. In sparse matrices, only a small proportion of a large, 2-D grid of numbers have values which are not zero. As a result, to save space in computer memory these sparse matrices are often stored in compressed form as lists or dictionaries, rather than as the full matrix of their original form. 

Sparsity is also encountered within biological systems. The brain is one example of a highly sparse system: at any given moment in time, it is estimated that [only 1%](https://aiimpacts.org/rate-of-neuron-firing/) of neurons in a human brain are in an 'active' state, in which they are sending out a voltage pulse or 'spike.' Given that on average, each neuron has [thousands of inputs](https://aiimpacts.org/scale-of-the-human-brain/), biological computation is highly sparse. 

One example of sparsity applied to create a neuromorphic system is event-based vision. These systems date back to the origin of the field when it was first [formally defined in the 80s by Carver Mead](https://ieeexplore.ieee.org/abstract/document/58356?casa_token=4dZiSoZykVYAAAAA:zhUavTNjZhXWQcPMe0aEFROYnAglDyfeTzNPKDiK1UN5JWPLqtudZj4USor5liiBXVCSEJaD). Traditional visual representations of the world around us often consist of static images taken many times per second: the analog version of this is traditional film, where a reel holds thousands of images which are shown quickly in succession to create the illusion of movement. Digital systems are similar, but instead of a physical reel of film, a matrix of color intensities stores a virtual copy of each image at each instant in time. This creates a 3-D structure representing video. However, much of the information in this 3-D structure is often redundant; many aspects of a scene change little throughout time. Video compression algorithms often focus on capturing only the differences between each frame of video. 

[Event-based vision](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9138762) takes a different view of visual representation inspired by the retina. These image sensors do not produce a continuous series of two-dimensional images; instead, each pixel produces an 'event' when it detects the intensity of light falling on it has changed above a certain threshold. As a result, the sensor can provide very quick 'reactions' to changes in a scene; instead of waiting for the entire image frame to be read out, the detected changes are sent out immediately. This allows event-based sensors to capture [very quick changes](https://www.youtube.com/watch?v=eomALySSGVU) efficiently. Another advantage these systems provide is the ability to capture very high dynamic ranges (environments which have both very bright and dark sections). These clear advantages have motivated a number of commercial ventures into event-based vision, from both start-ups such as [Prophesee](https://www.youtube.com/watch?v=MjX3z-6n3iA) and established companies such as [Samsung](https://www.youtube.com/watch?v=6xOmo7Ikwzk). However, the unfamiliar format of these sensors' visual data and relative lack of powerful software tools for them (e.g. Adobe Premiere for event-based data) may be a hurdle to more widespread adoption of these sensors, even as their cost decreases. 

Other sparse approaches to sensory encoding and data representation, including senses such as [smell](https://www.nature.com/articles/s42256-020-0159-4), [hearing](https://www.sciencedirect.com/science/article/pii/S0959438810000450?casa_token=Hl9E52r-UBAAAAAA:ZHTQNd97sGzLgN07J6OaF9t7F3NNDKAZW3was9eAEHTiZkLvx97lgy8wvSnL698uo2ybewc), and [touch](https://www.regenhealthsolutions.info/wp-content/uploads/2019/07/A-neuro-inspired-artificial-peripheral-nervous-system-for.pdf). Sparsity also provides a guiding principle for the design of neurmorphic algorithms, including [pattern recall, coding, and graph search](https://ieeexplore.ieee.org/document/9395703).

# Distribution

Compared to a computer chip, the human brain is remarkably resilient. Individual sections of the brain can be [damaged or removed](https://en.wikipedia.org/wiki/Epilepsy_surgery), but after a recovery period overall activity recovers and a patient can often return to a more normal lifestyle. In contrast, computers are fragile; making a small, random cut across a processor would almost certainly result in it failing completely. 

If you have a task which is very important to complete, you can take an alternate approach from executing it on one computer: instead, you can send the same task to multiple computers and examine the answers you get back. If the majority of the answers are the same, you can assume that answer is correct and use it. This way, even if individual parts of the system fail, the overall process can overcome those failures. This is a very simple method for what's known as 'distributed computing,' but the principles remain the same: using networks of components that pass messages, creating an overall system which can correctly carry out its task even when individual components fail. It's hypothesized that in many ways, the processing the brain carries out is distributed. Certain components (neurons and synapses) can fail, but the overall computation remains the same. 

Neuromorphic approaches often include a distributed approach to computing, both physically and conceptually. Redundant hardware 

An algorithm can distribute the same information multiple places, providing some protection against it becoming corrupted. 


# Novel Hardware



