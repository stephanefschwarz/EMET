# Translate texts in python

One of the most relevant problems in the field of Natural Language Process is how to deal with the different languages around the world. The simplest and fastest solution is to converge all the texts in just one language.

## How to use

First of all, install the requirements.

`pip install -r requirements.txt --user`

After that just execute the code:

> The code was written with Python 3

`python translate_post.py -f testList.txt -o outputFile.txt -l en`

Type `python translate_post.py --help` to see the documentation.

To use this module inside your application import the package and get an instance of TranslateTexts class, as shown in the below code.

```python
t = TranslateTexts()
t.translate_posts(['Wie geht es dir?', 'Tudo bem?'], dest='en')
print(t.translated_posts)
``` 