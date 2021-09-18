# categorial cross-entropy
def c2(input, target):

   _, labels = target.max(dim=1)

   return torch.nn.CrossEntropyLoss()(input, labels)
  
  # one-hot-encoding
def encodeOneHot(labels):
  ret = torch.FloatTensor(labels.shape[0], nl)
  ret.zero_()
  ret.scatter_(dim=1, index=labels.view(-1, 1), value=1)
  return ret

#if class = 0, then create vector 0s, if class = 1, draw from normal distribution  
def condition_to_latent_vec(conditions):
  latent_vecs = torch.zeros((conditions.shape[0], nz2))
  for i in range (conditions.shape[0]):
      if conditions[i] == 0:
          latent_vecs[i,:]= torch.zeros((1, nz2))
      else: 
          latent_vecs[i,:] = torch.randn((1, nz2))
            
  latent_vecs = latent_vecs.to(device)
    
  return latent_vecs
        
