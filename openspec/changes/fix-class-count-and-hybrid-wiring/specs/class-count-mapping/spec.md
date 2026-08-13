## ADDED Requirements

### Requirement: Classifier dimension derived from dataset class count
The system SHALL derive the number of classifier classes from the actual `ImageFolder` dataset (`len(dataset.classes)`) when building the dataset for the ImageNet dataset path, and MUST NOT hardcode a fixed count such as 1000 or 5. `build_loader()` MUST propagate this derived count to `config.MODEL.NUM_CLASSES` so `build_model()` constructs `DAMamba.head` with the correct output dimension.

#### Scenario: Five-class cauliflower dataset
- **WHEN** a dataset with 5 subdirectories in the train folder is built via `build_loader()`
- **THEN** `config.MODEL.NUM_CLASSES` equals 5 and `DAMamba.head.fc` has output dimension 5

#### Scenario: ImageNet-1k dataset
- **WHEN** an ImageNet-1k dataset with 1000 train subdirectories is built via `build_loader()`
- **THEN** `config.MODEL.NUM_CLASSES` equals 1000

#### Scenario: No hardcoded default substitution
- **WHEN** the dataset class count differs from the previous hardcoded value of 1000
- **THEN** the derived value (not 1000) is used without any warning or error

### Requirement: Pretrained classifier mismatch handled safely on load
The system SHALL keep name-level checkpoint validation intact. When the pretrained checkpoint classifier dimension differs from the model classifier dimension (e.g., a 1000-class pretrained head versus a 5-class model head), the incompatible head weights SHALL be filtered out of the loaded state dict so the model head stays at its initialization values, while all compatible backbone weights are loaded, and loading SHALL NOT raise an exception. Filtering SHALL be limited to shape-incompatible keys only.

#### Scenario: 1000-class checkpoint into 5-class model
- **WHEN** a checkpoint containing a 1000-class `head.fc` is loaded into a 5-class model
- **THEN** the `head.fc` keys are excluded from the load, `head.fc` remains at initialization, and the backbone weights are restored from the checkpoint without an exception

#### Scenario: Removing only incompatible keys
- **WHEN** a checkpoint contains a single shape-incompatible key and several compatible keys
- **THEN** only the shape-incompatible key is dropped and all compatible keys are loaded

#### Scenario: Matching classifier dimensions
- **WHEN** the checkpoint classifier dimension matches the model classifier dimension
- **THEN** all keys load successfully including the classifier head

### Requirement: Train, validation, and test class mappings remain consistent
The system SHALL use the same `ImageFolder` class-index mapping for train, validation, and test splits so that targets consistently lie within `[0, config.MODEL.NUM_CLASSES)`, and SHALL validate that the test split exists before referencing it.

#### Scenario: Consistent target ranges across splits
- **WHEN** train, validation, and test datasets are built from the same root hierarchy
- **THEN** every target in every split lies within `[0, config.MODEL.NUM_CLASSES)` where `NUM_CLASSES` equals `len(dataset.train.classes)`