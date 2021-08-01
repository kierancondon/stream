---
title: About
date: 2099-01-01
tags: All
---

# A simple web publishing system

This site is built using Streamlit, the open-source Python library. Streamlit makes it easy to create a front-end 
display of python code, and is especially good for creating web apps with user interface.

This particular app is an effort to share a simple web publishing system which can display both **text based and interactive code-based posts**.
Posts can be written in markdown or in pure python, including **front matter** to display a title, date, tags and contributing authors. 
Save your posts in a folder and then the app takes care of arranging and displaying all posts.

The site is hosted on Streamlit's own [sharing platform](https://streamlit.io/sharing) and the source code is [here on Github](https://github.com/kierancondon/stream). 
*(This site is also home to my own content)*

For quick note taking, [markdown](https://wordpress.com/support/markdown-quick-reference/) allows for fast writing and formatting. 
However, it lacks the ability to execute code and give a user interface. This is where .py files + streamlit really come in handy. Have a look at some 
examples here on the left and at the above Github repo.

## To write a post:

1. Create a new file (.md or .py) in the posts directory
2. Write the front matter at the top of the post.
* for markdown files:

```          
---
title: Your Title
date: 2022-01-01
tags: topic1, topic2
---
```

* for python files:

```          
"""
title: Your Title
date: 2022-01-01
tags: topic1, topic2
"""
```
3. Write the content
* for markdown files: just start writing in markdown after the front matter block!
* for python files: after the front matter block,
    1. Include this line of code: `Import streamlit as st`
    2. Define this display function within which you will write **everything else**: `def display():`
    3. To execute code, write python as normal
    4. To display text, wrap it in `st.write()`
    5. To display charts and other objects use streamlit functions such as `st.plotly_chart()`
    5. To display code whilst also executing it: proceed it by `with st.echo():`

4. After writing your post, save, then refresh the streamlit cache in the browser: press c, click "clear cache", then press r to reload.


## To use this system for your own local project, on your pc:

1. Create a new python project. Ensure to pip install streamlit in your python environment.
2. Create the sub-folder "posts" and copy the file "streamlit_app.py" from the above Github repository
3. Create your first .md file in the posts folder, including the metadata as described above
4. In your terminal, type "streamlit run streamlit_app.py" to launch the (local) site in your browser.

## To share your project online using streamlit sharing

1. Create a Github account and make a new blank repository. 
2. Add a requirements.txt file to your local project where you list any non-core python modules imported in your project
3. You may also want to add a Readme.md and a .gitignore file (optional)
3. Push your code to the new repo using the method described [here](https://www.codecademy.com/articles/push-to-github), 
[here](https://docs.github.com/en/github/importing-your-projects-to-github/importing-source-code-to-github/adding-an-existing-project-to-github-using-the-command-line), 
and [here](https://www.git-tower.com/learn/git/faq/push-to-github/)
3. Make a Streamlit Sharing account and follow the instructions to link to your github repo. Voila!