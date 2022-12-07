#! /bin/bash



python3 -m build
twine upload -u ${PYPI_USERNAME} -p ${PYPI_ACCESS_TOKEN} \
--repository-url https://upload.pypi.org/legacy/ dist/*