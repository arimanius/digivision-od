try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution('img2vec').version
except pkg_resources.DistributionNotFound:
    __version__ = '0.0.0'
