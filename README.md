# STEVE
STEVE will be a deep learning general purpose chatbot. Initially it will be only entertaining, the objective is just getting
it to answer something decent. Later on it will be an app, and after that it would be nice to make it do something actually 
useful for people. 

The model I am trying is, as far as I know, new, as I haven't seen it used anywhere. This might be explained by the fact that 
at the moment it doesn't work. The high level idea is to have an encoder and a decoder to get sentence level meaning vectors, 
and a neural network into which you feed these meaning vectors to get an answer's meaning vector. I like to call this the 'TSN'
as in Talking (should be used for chatbots, if it will ever work) Sandwich Network. RNN's are bread and feedforward nets
are the meat. 

Encoder: 
This is composed of two parts. A character level word encoder and a word level sentence encoder. The words in the input 
sentence are fed into the word encoder ,which is a character level RNN, one by one, and the hidden state is the word's meaning
vector. These word meaning vectors are concatenated and fed into the sentence level encoder which does the same thing. 
The hidden state of this RNN is our sentence meaning. 

Middle Layer: 
This is simply a feedforward fully connected network at the moment. It is what I am currently working on and trying to improve.
It receives a meaning vector from the encoder and outputs the answer's meaning vector. 

Decoder: 
This is an RNN that starts with the answer's meaning vector as initial hidden state and a start of sequence character as first 
input. The outputs at every timestep are the word level meaning vectors, which are individually fed to a word level decoder 
which works in the same exact way. The outputs from this are concatenated appended to a list. Each entry in the list is a
matrix of probabilities of characters for every word. 

Training: 
There are several options, of which I have only tried one. 

option 1:
The one I originally had in mind, which I have tried, consists in training the word encoder and decoder as an autoencoder, then
training the sentence encoder and decoder as an autoenocoder as well without updating the word encoder/decoder weights. Finally
feed sentence vectors into the middle layer and train that without updating the other networks. When training the middle layer
the target obviously isn't the input sentence as it was for the sentence autoencoder, but a valid asnwer. 
The word and sentence level autoencoders work, but training the middle layer doesn't produce any valid answer. 

option 2: 
Instead of training the word level autoencoder and the sentence level one separately, update their weights at the same time.
This way there will be no direct correspondence between word encodings (meaning that if you feed a word's vector into the word 
decoder you might not get the exact word out of it) but it should work at the sentence level. After doing that you train the 
middle layer the same way it is done in option1.

option 3:
train everything together directly on the final task. 

The loss can be on the final outputs, or in the first two cases it can also be the difference between the middle layer output
and the target meaning vector (which would not otherwise be calculated as you would use the character probabilities with the 
model output). 

Things I'm working on:
Implementing attention. 
Alternative models for the middle layer. 
