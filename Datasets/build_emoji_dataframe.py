# Commonly used
import numpy as np
import re
import pandas as pd
import os

# Packages used to scrap a website
import requests
import bs4

# One package used to detect emoji in text
import emoji


# Load the path where we cound find the 'emoji_dictionary.csv'
emoji_dictionary_path = r'XXXXX'


# This function is used to erase all the 'U+00' pattern in the twitter text
def remove_u_plus(text):
    result = re.sub(pattern=r'U\+00', repl=r'', string=text)
    return result


def show_emoji_in_tweet(text, emoji_dictionary):
    without_u = remove_u_plus(text)
    old_text = without_u
    old_text = old_text.encode('unicode_escape').decode('utf-8')
    result1 = re.sub(pattern='\\\\r', repl='', string=old_text)
    result2 = re.sub(pattern='\\\\n', repl='', string=result1)
    result3 = re.sub(pattern='\\\\x([a-z0-9]{2})', repl = '<\\1>', string=result2)
    old_text = result3
    for _, row in emoji_dictionary.iterrows():
        if row['R_Encoding'] in old_text:
            new_text = re.sub(pattern=row['R_Encoding'], repl=row['emoji'], string=old_text)
            old_text = new_text
        else:
            pass
        if row['R_Encoding_lower'] in old_text:
            new_text = re.sub(pattern=row['R_Encoding_lower'], repl=row['emoji'], string=old_text)
            old_text = new_text
        else:
            pass
    return old_text


def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI


def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False


if __name__ == '__main__':

    # Load the emoji data
    emoji_dictionary = pd.read_csv(os.path.join(emoji_dictionary_path, 'emoji_dictionary.csv'))
    emoji_dictionary = emoji_dictionary.drop(columns=['Number'])

    # construct a simple request to scratch the unicode and emoji from
    # http://www.unicode.org/emoji/charts/full-emoji-list.html
    res = requests.get('http://www.unicode.org/emoji/charts/full-emoji-list.html#1f469_200d_1f9b0')
    soup = bs4.BeautifulSoup(res.text, 'lxml')
    # construct the emoji unicode list
    unicode_list = []
    for i in soup.select('td.code'):
        unicode_list.append(i.text)
    emoji_list = []
    for j in soup.select('td.chars'):
        emoji_list.append(j.text)

    emoji_unicode = pd.DataFrame({'emoji': emoji_list, 'Codepoint': unicode_list})
    # merge two dataframes
    emoji_merged_file = pd.merge(emoji_dictionary, emoji_unicode, on='Codepoint')
    emoji_merged_file['R_Encoding_lower'] = emoji_merged_file.apply(lambda row: row['R_Encoding'].lower(),
                                                                    axis = 1)
    emoji_merged_file.to_pickle(os.path.join(emoji_dictionary_path, 'emoji_unicode_r_code.pkl'))

    # A simple case
    # Show the effectiveness of our show_emoji function
    text = '<ed><A0><BD><ed><B8><80>I love this moment!<ed><A0><BD><ed><B8><86><ed><A0><BD><ed><B8><84><ed><A0><BD><ed><B8><84><ed><A0><BD><ed><B8><84>'
    show_emoji_in_tweet(text, emoji_merged_file)

