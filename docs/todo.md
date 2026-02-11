# TODO

## Implment

### IEL1: Move FaceDetecxtionEngine into its own project

- New project in the solution
- Make it a .NET Standard console application or library that can be referenced by other projects in the solution.

This will allow us to use it in other projects and make it easier to maintain.

### IEL2: Add default registry of download URL locations for required models

Keep a list of searchable sites for finding standard models that are request by cleint extentions. This will allow us to easily find and download the required models for our project.

When the ctor that deosnt pass in custm URL is used, we can use the default registry to find the required model and download it if it is not already present on the system.

Use e.g. a JSON file to store the registry of download URL locations for required models. This will allow us to easily update the URLs without having to change the code.

For example include hugging face model URLs for the models we use in the project. Inlcude other commonly used model fecthing sites.

For instance the yolo11n.onxx model is used in the project shoudl be findable on hugging face and the registry should include the URL to download it from there.

### IEL3: Refactor project/solution structure to be .NET idiomatic


## Implemented
