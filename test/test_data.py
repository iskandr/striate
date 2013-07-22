from striate import data, util

def test_imagenet_loader():
  df = data.ImageNetDataProvider('/ssd/nn-data/imagenet/', 
                                 batch_range=range(1000), 
                                 category_range=range(20),
                                 batch_size=512)
  util.log('Index: %s', df.curr_batch_index)
  util.log('%s', df._get_next_batch()['data'].shape)
  util.log('Index: %s', df.curr_batch_index) 
  util.log('%s', df._get_next_batch()['data'].shape)
  util.log('Index: %s', df.curr_batch_index) 

if __name__ == '__main__':
  test_imagenet_loader()
