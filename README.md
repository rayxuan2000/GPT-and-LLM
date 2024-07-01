# GPT-and-LLM

## Introduction
In this project, I will implement a simplified version of GPT, which mainly use decoder structure to generate new text file.

## Diagram
- Structure that I use in this project:
 1. <img src="picture/1.jpg" alt="Head" width="300">

 2. <img src="picture/2.jpg" alt="Multihead" width="300">

 3. <img src="picture/3.jpg" alt="Forward Network" width="300">

 4. <img src="picture/4.jpg" alt="Block" width="300">

 5. <img src="picture/5.jpg" alt="Baby GPT" width="300">
 
## Notes
- LayerNorm and BatchNorm: Horizontal and Vertical difference, to put it simply
- Use one single character as token unit, which can be modified in other form.
- Generate mapping table from token to index and vice versa.
- We are concerned about text generation here, which means given some texts what would be the next token (character)? So the input to the GPT is some text and the output will be the next character. For example, given a sentence like "Your name is GPT-3",

```
when input is tensor([6])--'{'Y'}', the target is: [12]--'{'o'}'
when input is tensor([ 6, 12])--'{'Yo'}', the target is: [15]--'{'u'}'
when input is tensor([ 6, 12, 15])--'{'You'}', the target is: [13]--'{'r'}'
```

- For dataloader/batch generation, there are basically two parameters: _batch_size_ (how many independent sequences will we process in parallel) and _block_size_ (what is the maximum context length for predictions). It defines the shape of input (B,T) basically. For example if we generate three batch, we have the following,

```
when input is tensor([0])--' ', the target is: [9]--'i'
when input is tensor([0, 9])--' i', the target is: [14]--'s'
when input is tensor([ 0,  9, 14])--' is', the target is: [0]--' '
when input is tensor([ 0,  9, 14,  0])--' is ', the target is: [3]--'G'
when input is tensor([ 0,  9, 14,  0,  3])--' is G', the target is: [4]--'P'
when input is tensor([10])--'m', the target is: [8]--'e'
when input is tensor([10,  8])--'me', the target is: [0]--' '
when input is tensor([10,  8,  0])--'me ', the target is: [9]--'i'
when input is tensor([10,  8,  0,  9])--'me i', the target is: [14]--'s'
when input is tensor([10,  8,  0,  9, 14])--'me is', the target is: [0]--' '
when input is tensor([14])--'s', the target is: [0]--' '
when input is tensor([14,  0])--'s ', the target is: [3]--'G'
when input is tensor([14,  0,  3])--'s G', the target is: [4]--'P'
when input is tensor([14,  0,  3,  4])--'s GP', the target is: [5]--'T'
when input is tensor([14,  0,  3,  4,  5])--'s GPT', the target is: [1]--'-'
```

- In the first phase, just let the input go through a simple embedding layer (C,C) [C is vocabulary size]. When generating new text, a parameter _max_new_tokens_ denotes how many more characters it will generate. 

## Conclusion
