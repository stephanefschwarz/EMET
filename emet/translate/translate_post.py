# --- PACKAGE IMPORTATION ---

from googletrans import Translator
import argparse
import sys
import logging

logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# ---------- INSTANCE TO TRANSLATOR ---------- #

class TranslatorInstance:

	"""
	A class used to represent the Google Translator. This is a singleton classes used only by the _TranslateTexts_ class for translate Twetter posts.

	Attributes
	----------
	__api : Translator
		a singleton to Translator
	
	Methods
	-------
	get_translator_instance()
		get the a instance for the Translator API
	"""

	__api = None

	def __init__(self):

		"""Instantiate the Google translator API

		Parameters
		----------
		None

		Returns
		-------
		Nome
		"""
		
		if (TranslatorInstance.__api == None):

			TranslatorInstance.__api = Translator()


	def get_translator_instance():

		"""Get an instance of the Google Translator

		Parameters
		----------
		None

		Returns
		-------
		Translator
			an instance for Google Translator
		"""

		if (TranslatorInstance.__api == None):

			TranslatorInstance()

		return TranslatorInstance.__api

# ---------- TRANSLATE TEXTS ---------- #

class TranslateTexts(TranslatorInstance):

	"""
	Class used to effectively translate the list of texts.

	Attributes
	----------
	__api : Translator
		the singleton instance to Translator
	translated_posts : list
		a list of translated texts
	
	Methods
	-------
	translate_posts(posts)
		translates the list of passed twetter posts
	"""

	translated_posts = []
	__api = None

	def __init__(self):
		
		self.__api = TranslatorInstance.get_translator_instance()

	def translate_posts(self, posts, dest):

		"""Translate a list of Twetter publications

		Parameters
		----------
		posts : list
			The list of texts
		dest : str
			The output language

		Returns
		-------
		list
			a list of translated texts publication
		"""

		logging.info("Start translation...")

		progress = 0
		total_posts = len(posts)

		for post in posts:

			try:

				translated_text = self.__api.translate(post, dest=dest)

				TranslateTexts.translated_posts.append(translated_text.text)

			except Exception as e:

				logging.error('Could not translate this text: %s', post)

				TranslateTexts.translated_posts.append(post)
				

			progress = progress + 1

			percentage = round((progress / total_posts) * 100, 2)
			output_print = "{}% | {}/{}".format(percentage, progress, total_posts)
			
			logging.info(output_print)

			# Poor way to show a progress bar :|			
			sys.stdout.write("\r {:<70}".format(output_print))
			sys.stdout.flush()

		logging.info("DONE!")


# ---------- METHODS FOR FILE ---------- #

def command_line_parsing():
	"""Parse command lines

		Parameters
		----------
		file_path : str
			file path to the list of text to be translated
		output_file_path : str
			file to store the output file

		Returns
		-------
		parser
			The arguments from command line
	"""	
	logging.info("Parsing command line")
	
	parser = argparse.ArgumentParser(description = __doc__)

	parser.add_argument('--file-path', '-f', 
						dest='file_path', 
						required=True,
						help='File path to the list of text to be translated.')

	parser.add_argument('--output-file-path', '-o', 
						dest='output_file_path', 
						required=True,
						help='Output path to store the translated texts.')

	parser.add_argument('--dest-language', '-l', 
						dest='language', 
						required=False,
						default='en',
						help='The output language, by default english.')

	return parser.parse_args()

def read_text_file(file_path):

	logging.info("Reading file.")

	"""Read file from path

		Parameters
		----------
		file_path : str
			file path to the list of text to be translated
		
		Returns
		-------
		list
			List of texts
	"""	

	with open(file_path) as file:

		texts = file.read().splitlines()

	file.close()
	
	return texts

def write_translated_file(output_path, translated_list):

	logging.info("Writing result.")

	"""Write the generated file

		Parameters
		----------
		output_path : str
			The path to write the output file
		
		Returns
		-------
		None
	"""

	file = open(output_path, 'w')

	file.write("\n".join(translated_list))

	file.close()


if __name__ == '__main__':

	args = command_line_parsing()
	
	texts = read_text_file(args.file_path)

	logging.info("Getting Google Translator Instance.")

	t = TranslateTexts()

	t.translate_posts(texts, args.language)

	write_translated_file(args.output_file_path, t.translated_posts)

	print("\nDONE!")

	# ---------- SIMPLEST WAY ---------- #
	
	# t = TranslateTexts()
	# t.translate_posts(['Wie geht es dir?', 'Tudo bem?'], dest='en')
	# print(t.translated_posts)