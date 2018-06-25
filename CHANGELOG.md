# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


## [0.0.17] - 2018-06-24

### Added
- Contributions file.

### Changed
- Moved to MIT license.
- Fixed test/train type evaluation classifier bad results.  Note that you'll
  want to rerun your evaluations and retrain your models if you used the
  `train-test` method of evaluation.  For single pass cross validation (default
  validation type) you're fine.


## [0.0.16] - 2018-02-21

### Changed
- Use zenbuild as submodule.

### Removed
- IO utility library moved to `clj-actioncli`, which this project already
  includes.


## [0.0.15] - 2017-10-17
### Changed
- Split out serialization utility functions.
- Write model bug fix.

## [0.0.14] - 2017-08-21
### Changed
- Nail down the SVMlib to a version rather than range given in POM.


## [0.0.13] - 2017-06-10
### Changed
- Fixed model write/output bugs and added logging.

### Added
- Output matrix prediction file.


## [0.0.12] - 2017-05-16
### Changed
- Performance metric repoting bug fixes.
- Report class label on performance metrics output.

### Added
- No longer require numeric IDs for two pass cross validation.
- Classifier creation checking.


## [0.0.11] - 2017-05-08
### Changed
- Persisted model data usage bug fix.
- More checking and logging


## [0.0.10] - 2017-04-27
### Changed
- Weka NLP bug fix.


## [0.0.9] - 2017-04-27
### Changed
- Configurable precision in spreadsheet results.
- Upgrade from Weka 3.6 to 3.8
- Fix two pass cross fold validation.

## Added
- Create predictions spreadsheet output (CSV and Excel); nice for R analysis.


## [0.0.8] - 2016-12-14
### Added
- Added support for no attribute word vector (text) models.
- Added more word count options


## [0.0.7] - 2016-11-09
### Changed
- Moving combined feature creation to [feature](https://github.com/plandes/clj-nlp-feature) repository.
- API documentation creation version bump.


[Unreleased]: https://github.com/plandes/clj-ml-model/compare/v0.0.17...HEAD
[0.0.17]: https://github.com/plandes/clj-ml-model/compare/v0.0.16...v0.0.17
[0.0.16]: https://github.com/plandes/clj-ml-model/compare/v0.0.15...v0.0.16
[0.0.15]: https://github.com/plandes/clj-ml-model/compare/v0.0.14...v0.0.15
[0.0.14]: https://github.com/plandes/clj-ml-model/compare/v0.0.13...v0.0.14
[0.0.13]: https://github.com/plandes/clj-ml-model/compare/v0.0.12...v0.0.13
[0.0.12]: https://github.com/plandes/clj-ml-model/compare/v0.0.11...v0.0.12
[0.0.11]: https://github.com/plandes/clj-ml-model/compare/v0.0.10...v0.0.11
[0.0.10]: https://github.com/plandes/clj-ml-model/compare/v0.0.9...v0.0.10
[0.0.9]: https://github.com/plandes/clj-ml-model/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/plandes/clj-ml-model/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/plandes/clj-ml-model/compare/v0.0.6...v0.0.7
