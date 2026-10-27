"""Methods written once for every model (ENG-E8): they read the model only through its adapter.

Outside the disk tier's fingerprint (`engine/disk_cache._engine_files` leaves this package out until the
arithmetic manifest of ENG-243 lands): a method chooses which rows a model computes, never how.
"""
