# test the point cloud classes
import importlib

import pytest


class TestImport:
    def test_install(self):
        quickshift = importlib.import_module(
            "forest_inventory_pipeline.cluster.quickshiftpp"
        )
        assert quickshift is not None
