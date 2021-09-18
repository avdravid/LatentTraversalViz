from torch.autograd import Variable

init_noisez1 = Variable(torch.zeros(1, nz1).to(device), requires_grad = True)

init_noisez2 = Variable(torch.randn(1, nz2).to(device), requires_grad = True)

optim = torch.optim.Adam([init_noisez1, init_noisez2], lr=0.01, betas=(0.5, 0.999))  

netG.eval()
Classifier.eval()

c1 = torch.nn.BCELoss()

# get softmax probability of COVID positive on real image(s)
clf, _ = Classifier(original_image.to(device))
clf = clf[0,1]


for epoch in range(0,5000):
    
    optim.zero_grad()
    sample = netG(init_noisez1,init_noisez2).to(device)
    sample = (sample.reshape([-1,3,128,128]))
    result,_ = Classifier(sample) # get classifier output on generated sample(s)
    prob = result[0,1]
    
    class_loss = c1(prob, clf) #matching softmax probability of COVID-positive for both real and fake sample
   
    original_image =  (original_image.reshape([1,3,128,128]))
    
    loss =  class_loss + 4*torch.mean((original_image - sample)**2)
    
    print("E:", epoch+1, "loss:", loss.item())
    loss.backward()
    optim.step()
    
    # visualize image(s)
    if (epoch+1) % 100 == 0:
        reconstructed_image = netG(init_noisez1, init_noisez2).detach().cpu().view(-1, 3, 128, 128)
        
        reconstructed_image = reconstructed_image[0,]
        
      
        fig=plt.figure(figsize=(5, 5))
        plt.title('Reconstruction')
        plt.axis('off')

        minifig= fig.add_subplot(1, 2, 1)
        minifig.axis('off')
        minifig.title.set_text('Original' + "\n")
        original_image = original_image.cpu().view(3, 128, 128)
        original_image = (np.transpose(original_image,(1,2,0))+1)/2
        original_image = (original_image)
        plt.imshow(original_image)
        
        minifig= fig.add_subplot(1, 2, 2)
        minifig.title.set_text('Reconstructed')
        minifig.axis('off')
        reconstructed_image = np.transpose(reconstructed_image,(1,2,0))
        reconstructed_image = (reconstructed_image + 1)/2
        plt.imshow(reconstructed_image)

        plt.show()
