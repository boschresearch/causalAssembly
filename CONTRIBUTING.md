# Contributing

Contributions are welcome and appreciated!
Every little bit helps, and credit will always be given.

When contributing to causalAssembly (such as code, bugs, documentation, etc.) you
agree to the Developer [Certificate of Origin](http://developercertificate.org/)
and the causalAssembly license (see the [LICENSE](https://github.com/bosch-cc-mfd/ProductionLineGenerator/LICENSE.txt) file).

For commits, it is best to simply add a line like this to your commit message,
with your name and email

    Signed-off-by: Jane Doe <developer@example.com>

Please try to write a good commit message, see [good commit message wiki](https://github.com/nexB/aboutcode/wiki/Writing-good-commit-messages) for
details. In particular use the imperative for your commit subject: think that
you are giving an order to the codebase to update itself.

## Feature requests and feedback

To send feedback or ask a question, [file an issue](https://github.com/bosch-cc-mfd/ProductionLineGenerator/issues)

If you are proposing a feature:

* Explain how it would work.
* Keep the scope as simple as possible to make it easier to implement.
* Remember that your contributions are welcome!

## Bug reports

When `reporting a bug` please include:

* Your operating system name, version and architecture (32 or 64 bits).
* Your Python version.
* Your causalAssembly version.
* Any additional details about your local setup that might be helpful to
  diagnose this bug.
* Detailed steps to reproduce the bug, such as the commands you ran and a link
  to the code you are scanning.
* The errors messages or failure trace if any.
* If helpful, you can add a screenshot as an issue attachment when relevant or
  some extra file as a link to a [Gist](https://gist.github.com).

## Documentation improvements

Documentation can come in the form of new documentation pages/sections, tutorials/how-to documents,
any other general upgrades, etc. Even a minor typo fix is welcome.

If something is missing in the documentation or confusing,
please file an issue with your suggestions for improvement.
Your help and contributions makes causalAssembly docs better, we love hearing from you!

The causalAssembly documentation is hosted at [cuddly-robot-g34n2v1.pages.github.io/](https://cuddly-robot-g34n2v1.pages.github.io/).

## Development

To set up causalAssembly for local development:

1. Fork causalAssembly on GitHub, click [fork](https://github.com/bosch-cc-mfd/ProductionLineGenerator/fork) button

2. Clone your fork locally.

3. Create a branch for local development:

   ```bash
   git checkout -b name-of-your-bugfix-or-feature
   ```

4. To configure your local environment for development, locate to the main
   directory of the local repository, run

   ```bash
   make sync-venv
   ```
   All code must be PEP8 compatible. We have set up checking code quality with [pre-commit](https://pre-commit.com/) which runs [ruff](https://github.com/charliermarsh/ruff), a Python linter written in Rust.
   It is recommended to install [pre-commit](https://pre-commit.com/) locally to check code quality on every commit.

5. Now you can make your code changes in your local clone.
   Please create new tests for your code. We love tests!

6. When you are done with your changes, run all the tests.
   Use this command:
   ```bash
   python -m pytest
   ```

7. Check the status of your local repository before commit, regarding files changed:
   ```bash
   git status
   ```

8. Commit your changes and push your branch to your GitHub fork::
   ```bash
   git add <file-changed-1> <file-changed-2> <file-changed-3>
   git commit -m "Your detailed description of your changes." --signoff
   git push <repository-alias-name> name-of-your-bugfix-or-feature
   ```

9. Submit a pull request through the GitHub website for this branch.

### Pull Request Guidelines

If you need a code review or feedback while you are developing the code just
create a pull request. You can add new commits to your branch as needed.

For merging, your request would need to:

1. Include pytests that are passing (run `python -m pytest`).
2. Update documentation as needed for new API, functionality etc (run `mkdocs build`).
3. Add a note to `CHANGELOG.md` about the changes.
4. Add your name to the `Authors` section in the [README](https://github.com/bosch-cc-mfd/ProductionLineGenerator/blob/main/README.md).
