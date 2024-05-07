# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("./Final_Pulse_Data.csv")

mask = data == 'Did not report'
data = data.mask(mask, np.nan)

# Plot One
y = 'Provider_Of_Free_Groceries'
x = 'Enough Food, but not always the kinds wanted'
sns.barplot(x=x, y=y, data=data)
# # Add labels and a title
plt.title('Lack of Client Choice from Food Providers')
plt.xlabel("Choice of Food Not Wanted")
plt.ylabel("Food Provider")
# Display the plot
plt.show()
