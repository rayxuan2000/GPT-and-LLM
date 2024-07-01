# GPT-and-LLM

## Introduction
In this project, I will implement a simplified version of GPT, which mainly use decoder structure to generate new text file.

## Notes
- LayerNorm and BatchNorm: Horizontal and Vertical difference, to put it simply
  
- Structure that I use in this project:
 1. <img src="picture/1.jpg" alt="Head" width="300">

 2. <img src="picture/2.jpg" alt="Multihead" width="300">

 3. <img src="picture/3.jpg" alt="Forward Network" width="300">

 4. <img src="picture/4.jpg" alt="Block" width="300">

 5. <img src="picture/5.jpg" alt="Baby GPT" width="300">

- Use one single character as token unit, which can be modified in other form.
- Generate mapping table from token to index and vice versa.
- We are concerned about text generation here, which means given some texts what would be the next token (character)? So the input to the GPT is some text and the output will be the next character. For example, given a sentence like "Your name is GPT-3",

```
when input is tensor([6])--'{'Y'}', the target is: [12]--'{'o'}'
when input is tensor([ 6, 12])--'{'Yo'}', the target is: [15]--'{'u'}'
when input is tensor([ 6, 12, 15])--'{'You'}', the target is: [13]--'{'r'}'
```

## Conclusion
