# Generating Memorable Mnemonic Encodings of Numbers
This project uses natural language processing to generate memorable encodings of numbers using the [major system](https://en.wikipedia.org/wiki/Mnemonic_major_system). For example, you can encode the number "86101521" as the phrase "officiate wasteland", which is easier to remember.

To generate an mnemonic encoding of your own, run `major_system.py`. The `main` method encodes four example numbers, which you can change to your own numbers. The method also shows how to use other encoders we created. To run this code, you need Python 3.x with `nltk` and `numpy` installed.

You can find more details about this project in our paper here: [*Generating Memorable Mnemonic Encodings of Numbers*](https://arxiv.org/pdf/1705.02700.pdf). This paper includes a review of prior work, an explanation of several encoders we tried, and a password memorability study using our encoder. We found that a model combining part-of-speech sentence templates with an n-gram language model produces the most memorable password representations.
