
from bs4 import BeautifulSoup
import pandas as pd
import time
import requests
import json
import sys
import os


def get_jsons(directory):
  """
  Helper function for getting json files in a directory
  :param directory: absolute path to directory containing the jsons
  :return: list of jsons
  """
  f = []
  for root, _, files_ in os.walk(directory):
      for file in files_:
          if file.endswith('.json'):
              f.append(os.path.join(root, file))
          else:
              continue
  return f


def grab_text(tag='p', soup_obj=None):
  """
  Helper function to grab text from paragraph tags in the beautiful soup object
  :param tag: str; html tag to search for
  :param soup_obj: BeautifulSoup type object
  :return:
  """
  if not soup_obj:
      print('Please pass a BeautifulSoup object for parameter soup_obj')
      raise TypeError
  tmp = []
  try:
      for entry in soup_obj.findAll(tag):
          tmp.append(entry.get_text())
  except AttributeError:
      print('Error occured {}'.format(sys.exc_info()[0]))
  par_no = 1
  par_dict = {}
  for para in tmp:
      par_dict['paragraph_{}'.format(par_no)] = para
  return par_dict


class Textract:

  def __init__(self, sarc_dir=None, files=None):

      self.directory = sarc_dir  # path to directory
      self.files = files  # list like
      self.data = None

      if not files:
          if not sarc_dir:
              print('Expecting either a path to a directory of files or list of files to analyze')
              raise SyntaxError
          else:
              self.files = get_jsons(sarc_dir)

  def get_data(self):
      """
      Get list of jsons from json files
      :return: None
      """
      tmp = []
      for file in self.files:
          with open(file.encode('unicode_escape')) as f:
              for line in f:
                  tmp.append(json.loads(line))
      self.data = tmp
      return None

  def ping_pages(self, max_pings):
      """
      Pings all urls in self.data and extract pargraph text
      :return: list of dicts
      """
      ind = 0
      for e in self.data:
          try:
              res = requests.get(e['article_link'])
              if res.status_code == 200:
                  soup = BeautifulSoup(res.content, 'html.parser')
                  self.data[ind]['paragraph_text'] = grab_text(soup_obj=soup)
              else:
                  continue
          except:
              print('Error occured {}'.format(sys.exc_info()[0]))
              continue
          ind += 1
          print('Pages Sucessfully scraped: {}'.format(ind + 1))
          if max_pings -1 < ind:
              print("Max page threshold met. Halting now")
              break
      return None
