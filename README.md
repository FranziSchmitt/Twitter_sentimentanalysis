# Twitter_sentimentanalysis
This is the portforlio project, I worked on while I participated in Data Science Retreat in Berlin. 

I set up a Raspberry Pi and scraped for Tweets that mention at least one of the German parties. 

In a first attempt to find patterns, I tried k-means clustering of the tweets. However, using Latent Dirichlet Allocation to find topics proved to be the better way to go.
The quality of the topics heavily depends on the quality of the preprocessing. Since tweets, in all their shortness, contain heaps of noise, the right amount of text preprocessing was key. 

In addition to finding topics, I also looked at topics over time and could see that topic popularity fits political events taking place in Germany.
