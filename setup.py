from setuptools import setup

setup(name='tf_sentence_transformers',
      version='0.1',
      description='A Tensorflow SentenceTransformer layer that takes in strings as input.',
      url='https://github.com/zzsi/tf_sentence_transformers',
      author='ZZ Si',
      author_email='zhangzhang.si@gmail.com',
      license='MIT',
      packages=['tf_sentence_transformers'],
      install_requires=[
          'tensorflow>=2.0.0',
          'transformers'
      ],
      include_package_data=True,
      zip_safe=False)