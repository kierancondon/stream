import streamlit as st
import os
import importlib

@st.cache()
def get_post_front_matter():
    """
    Open files in the folder 'posts' and gather all front matter (title, date, tags...) and file info for each post
    :return:
        all_front_matter:           dictionary with key = post tile and value = dictionary of front matter, filename...
        all_tags:                   dictionary with key = tag and value = list of post titles bearing this tag
        date_title_to_title:        dictionary to help processing titles and date_titles
        title_to_date_title         dictionary to help processing titles and date_titles
    """

    short_filenames = os.listdir('posts')
    short_filenames = [x for x in short_filenames if x not in ['__init__.py', '__pycache__']]
    filenames = [os.path.join('posts', x) for x in short_filenames]

    all_front_matter = {}

    for s, f in zip(short_filenames, filenames):

        fm_marker = '---' if f[-2:] == "md" else '"""'

        front_matter = {'filename': f}
        front_matter['short_filename'] = s[:-3]
        front_matter['fm_marker'] = fm_marker
        front_matter['file_type'] = f[-2:]

        count_bars = 0
        with open(f, 'r') as myfile:
            for line in myfile:
                line = line.rstrip('\n')
                if line == fm_marker and count_bars == 1:
                    break
                elif count_bars == 1:
                    front_matter[line[:line.find(':')]] = line[line.find(':')+1:].lstrip()
                elif line == fm_marker:
                    count_bars += 1
        front_matter['tags'] = front_matter['tags'].split(', ')

        all_front_matter[front_matter['title']] = front_matter

    # Add some useful dictionaries
    all_tags = {}
    date_title_to_title = {}
    title_to_date_title = {}
    for title, front_matter in all_front_matter.items():

        #all_tags
        for tag in front_matter['tags']:
            tag = tag.lower().capitalize()
            if tag not in all_tags.keys():
                all_tags[tag] = [title]
            else:
                all_tags[tag].append(title)

        #date_title
        date_title_to_title[front_matter['date'] + " | " + title] = title
        title_to_date_title[title] = front_matter['date'] + " | " + title

    all_tags['All'] = list(all_front_matter.keys())

    return all_front_matter, all_tags, date_title_to_title, title_to_date_title

all_front_matter, all_tags, date_title_to_title, title_to_date_title = get_post_front_matter()


def retrieve_post_content_after_end_of_front_matter(string, substring):
    """
    :param string:          All text inside the post's .md or .py
    :param substring:       e.g. "---" for .md
    :return:                string = all text after the front matter i.e. the content of the post
    """
    end_of_front = string.find(substring, string.find(substring) + 1) + 3
    return string[end_of_front:]


# BUILD SIDEBAR

st.sidebar.header("Select a Post:")

tag_selector = st.sidebar.selectbox('Tags to display:', sorted(all_tags.keys()), index=0)
display_dates = st.sidebar.checkbox('Display dates', value=False)

if display_dates == False:
    selected_post = st.sidebar.radio('Selected', sorted(all_tags[tag_selector], reverse=False))

else:
    selected_post = st.sidebar.radio('Selected', sorted([title_to_date_title[t] for t in all_tags[tag_selector]], reverse=True))
    selected_post = date_title_to_title[selected_post]


# BUILD MAIN SECTION DISPLAYING ONE POST AT A TIME

post_info = all_front_matter[selected_post]

f = post_info['filename']
with open(f, 'r') as myfile:
    post_data = myfile.read()
    post_data = retrieve_post_content_after_end_of_front_matter(post_data, post_info['fm_marker'])


# Date and contributor

if 'contributor' in post_info.keys():
    contributor = post_info['contributor']
    contributor = f' by ***{contributor}***'
else:
    contributor = ''

date = post_info['date']
if date != '2099-01-01':
    st.markdown(f'*Written on * ***{date}***' + contributor)


# Publish content
if post_info['file_type'] == 'md':
    st.write(post_data)
elif post_info['file_type'] == 'py':
    post_module = importlib.import_module('posts.' + post_info['short_filename'])
    #post_module = sys.modules['posts.' + post_info['short_filename']]
    result = getattr(post_module, 'display')()