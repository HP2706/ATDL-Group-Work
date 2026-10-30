"""Compatibility entrypoint for the shared baseline/native mesh analyzer."""
from video3d.mesh_analysis import analyze, cache_visibility, closest_surface, visibility

__all__ = ['analyze', 'cache_visibility', 'closest_surface', 'visibility']
