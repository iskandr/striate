import cudaconv2

# driver.init()
# device_info = (0, 0)
# for i in range(driver.Device.count()):
#  dev = driver.Device(i)
#  ctx = dev.make_context()
#  ctx.push()
#  free, total = driver.mem_get_info()
#  print 'Free Memory for Device', i, 'is', free / 1000000, 'MB'
#
#  if device_info[1] < free:
#    device_info = (i, free)
#
#  ctx.pop()
#  ctx.detach()

# print 'Choose Device', device_info[0]
# dev = driver.Device(device_info[0])

CONTEXT = cudaconv2.init()
