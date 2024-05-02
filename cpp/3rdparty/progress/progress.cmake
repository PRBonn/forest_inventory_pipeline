include(FetchContent)
FetchContent_Declare(progress GIT_REPOSITORY https://github.com/mehermvr/progress.git
                     GIT_TAG 5a3cd035105d3d33b6ff5f587dcf4d3b6c3fda3d SYSTEM EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(progress)
