# OpenABM-Covid19 documentation website

This is the source code repository for the [OpenABM-Covid19 website](https://openabm-covid19.bdi-pathogens.github.io/).

## Creating a website with MKdocs tutorial

There's a great [tutorial on YouTube](https://www.youtube.com/watch?v=Q-YA_dA8C20) on how to create
your own website like this one using mkdocs.

## Notes on MkDocs format

WARNING: You MUST use relative links AND INCLUDE the trailing `.md` extension. Otherwise
the links may work in your local dev environment but fail when deployed to the GitHub Pages
website. See https://www.mkdocs.org/user-guide/writing-your-docs/#internal-links for further info.

## Mathematical formulae

There's a [guide on adding math formulae](https://squidfunk.github.io/mkdocs-material/reference/math/#usage) available.
The `MathJax` option has been enabled on this website. See the main website index file for an example.


## Setting up your build environment

Note: Ensure you know how to set up PyEnv and PyEnv-Virtualenv before you start. E.g. by reading: https://akrabat.com/creating-virtual-environments-with-pyenv/ 

```sh
brew install pyenv pyenv-virtualenv
pyenv virtualenv 3.10.7 mywebsite
pip install mkdocs mkdocs-material pillow cairosvg
```

## Generating and previewing documentation locally

Simply execute the following (assuming you are in the mywebsite virtual env):-

```sh
cd jpg-lab
pyenv local mywebsite
mkdocs serve
```

Click the link in the terminal to view the local build of the documentation.

## Building and publishing

The website is automatically published when pushed to the main site.
This is achieved using GitHub Actions. See [the action definition](.github/workflows/ci.yml) for details.