<h1 align="center">
    Repository title
</h1>

<p align="center">
:warning: The repository title might also be different from that of the paper.
</p>

<p align="center"><img src="assets/image.png" alt=""/></p>

<p align="center">
:warning: The image above is just a placeholder. We do not need to use the HSP logo.
</p>

<h4 align="center">
  Paper Title
</h4>

<div align="center">
  Journal, vol. X, no. y, pp. abc-def, Month Year
</div>

<div align="center">
  <a href=""><b>Paper</b></a> |
  <a href=""><b>arXiv</b></a> |
  <a href=""><b>Video</b></a> |
</div>

<div align="center">
:warning: Other kind of contents are welcome here.
</div>

<img src="assets/fake_badge.png" alt=""/></p>
:warning: The image above is just a placeholder for possible badges the user might want to insert in the repository. Badges can point to DOIs, Continuous Integration status, Code coverage, etc.


## Table of Contents

- [Update](#updates)
- [Installation](#installation)
- [Reproduce the results](#reproduce-the-paper-results)
- [Run the code with custom data](#run-the-code-with-custom-data-optional)
- [License](#license)
- [Citing this paper](#citing-this-paper)

## Updates

YYYY-MM-DD - Added ABC

...

YYYY-MM-DD - Added XYZ

:warning: Use this section to communicate possible updates, such as bug fixes, introduction of new features, extension of the results to other datasets, implementation of new algorithms.

:warning: If new algorithms are implemented, these might be **also connected to several papers representing the evolution of your research**. In that case, it could be useful to define GitHub `tags` for each paper so that the user could easily navigate the history of your releases, one per paper. Having software releases and/or a GitHub packages associated to those tags is also welcome.
Tags connected to papers might also be added in the above "Updates" section for ease of navigation and to distinguish them from other possible tags not connected to papers.

## Installation

```console
<all the instructions require to install your software>
```
:warning: In this section it would be good to provide the user with all the information necessary to install the correct dependencies, with the correct versions. If the dependencies are built from sources, indicating a specific commit/tag represents an alternative to specifying the version.

### Execution inside a container (alternative)

If the user provides an image loaded inside the GitHub Container Registry:

```console
    docker pull ghcr.io/hsp-iit/repository-title:latest
    docker run <all yuor parameters> ghcr.io/hsp-iit/repository-title:latest
```
:warning: We still have not prepared instructions on how to load an image inside the GitHub Container Registry.

If the user only provides a recipe:

```console
    cd Dockerfiles
    docker build -t repository-title:latest
    docker run <all yuor parameters> repository-title:latest
```

## Reproduce the paper results

Before running the experiments, it is suggested to run the following sanity checks to make sure that the environment is properly configure:

```console
<all the instructions required to check that the environent has been configured properly>
```

Instructions for reproducing the experiments:

```console
<all the instructions required to reproduce the results>
```

Adding an example of the expected outcome might be useful.

## Run the code with custom data (optional)

Adding information on the structure of the input data and how it gets processed might be useful.

```console
<all the instructions required to run your code on custom data>
```

## License

Information about the license.

:warning: Please read [these](https://github.com/hsp-iit/organization/tree/master/licenses) instructions on how to license HSP code.

## Citing this paper

```bibtex
@ARTICLE{9568706,
author={Author A, ..., Author Z},
journal={Journal},
title={Title},
year={Year},
volume={X},
number={y},
pages={abc-def},
doi={DOI}
}
```

## Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="assets/image.png" width="40">](https://github.com/hsp-iit) | [@hsp-iit](https://github.com/hsp-iit) |

:warning: The image above is just a placeholder. We do not need to use the HSP logo.
