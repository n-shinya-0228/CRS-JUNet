import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from lib.utils.raw_point_copy_paste_e0c import DonorDatabase, SCHEMA, quality
from lib.utils.pointnet_copy_paste_e0 import sha256


class DonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping=yaml.safe_load(Path("config/labels/semantic-kitti.yaml").read_text())["learning_map"]

    def person(self):
        rng=np.random.default_rng(3)
        xyz=rng.uniform([-.3,-.2,-.8],[.3,.2,.8],(300,3)).astype(np.float32)
        return np.column_stack((xyz,np.full(300,.4))).astype(np.float32),np.full(300,(12<<16)|30,np.uint32)

    def test_incomplete_database_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"provenance.json").write_text(json.dumps(dict(schema=SCHEMA,complete=False)))
            with self.assertRaisesRegex(ValueError,"incomplete"):
                DonorDatabase(root,self.mapping)

    def test_validation_donor_rejected_at_manifest_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"provenance.json").write_text(json.dumps(dict(schema=SCHEMA,complete=True)))
            (root/"manifest.json").write_text(json.dumps([dict(source_sequence="08")]))
            with self.assertRaisesRegex(ValueError,"not_train"):
                DonorDatabase(root,self.mapping)

    def test_missing_primary_class_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"provenance.json").write_text(json.dumps(dict(schema=SCHEMA,complete=True)))
            (root/"manifest.json").write_text("[]")
            with self.assertRaisesRegex(ValueError,"missing_primary"):
                DonorDatabase(root,self.mapping)

    def test_checksum_and_packed_correspondence_checked_on_each_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            points,packed=self.person()
            patch,center,check=quality(points,packed,self.mapping)
            path=root/"donor.npz"
            np.savez_compressed(path,xyz_original=points[:,:3],xyz_relative=patch[:,:3],remission=points[:,3],
                                packed_labels=packed,raw_semantic=packed&65535,original_center=center,
                                source_sequence="00",source_frame="000123",schema_version=SCHEMA)
            row=dict(source_sequence="00",source_frame="000123",source_instance_id=12,file=path.name,
                     point_count=len(points),sha256=sha256(path))
            db=DonorDatabase.__new__(DonorDatabase);db.root=root;db.learning_map=self.mapping
            np.testing.assert_array_equal(db.load(row),patch)
            row["source_instance_id"]=13
            with self.assertRaisesRegex(ValueError,"correspondence|content_or_provenance"):
                db.load(row)
            row["source_instance_id"]=12
            row["sha256"]="wrong"
            with self.assertRaisesRegex(ValueError,"hash_mismatch"):
                db.load(row)
            row["file"]="../outside.npz"
            with self.assertRaisesRegex(ValueError,"path_or_hash"):
                db.load(row)

    def test_quality_rejects_mixed_instances_and_outliers_without_deleting_points(self):
        points,packed=self.person()
        packed[0]=np.uint32((13<<16)|30)
        patch,_,check=quality(points,packed,self.mapping)
        self.assertEqual(len(patch),len(points))
        self.assertIn("invalid_or_mixed_instance",check["reasons"])
        points[:30,:3]+=10
        packed[:]=(12<<16)|30
        patch,_,check=quality(points,packed,self.mapping)
        self.assertEqual(len(patch),len(points))
        self.assertTrue(check["reasons"])


if __name__=="__main__":
    unittest.main()
