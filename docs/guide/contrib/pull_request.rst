
*********************
Submit a Pull Request
*********************

.. contents::

Synchronize Version
===================

This is a reference to submit a CVM pull request.

- Before submit or development, please synchronize your code on the most recent version of your branch, you can do it by

  .. code-block:: bash

    git checkout [your branch]
    git pull

- Also, please keep up with the [base branch] and merge if needed by

  .. code-block:: bash

    git merge origin/[master branch]

- If conflicts occur after merge, fix them before submit.

.. _code_formating:

- After development, make sure code style check pass by typing the following command, and all the existing test-cases pass.

  .. code-block:: bash

    # Reformating python modules
    pylint [your python file]


Write Tests and Documentation
=============================

- Add test-cases to cover the new features or bugfix the patch introduces, the locations of test files are enumerated as follows:

.. list-table::
   :widths: 25 25

   * - Module Name
     - Test File Directory
   * - cvm-runtime
     - ./test
   * - CVM python interface 
     - ./test/python
   * - MRT
     - ./test/mrt

- Document the code you wrote, see more at :doc:`Write Documentation and Tutorials <write_document>`


Create Pull Requests
====================

- Push your code to the remote repo by:

  .. code-block:: bash

    # Set upstream if it's first time to push [your branch]
    git push --set-upstream origin [your branch]
    git push

- Send the pull request and fix the problems reported by automatic checks (if existed).
- Request code reviews from other contributors and improves your patch according to feedbacks.

  - To get your code reviewed quickly, we encourage you to help review others' code so they can do the favor in return.
  - Code review is a shepherding process that helps to improve contributor's code quality.
    We should treat it proactively, to improve the code as much as possible before the review.
    We highly value patches that can get in without extensive reviews.
  - The detailed guidelines and summarizes useful lessons.

- The patch can be merged after the reviewers approve the pull request.

