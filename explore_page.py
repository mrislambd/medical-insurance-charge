import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def data_load():
    df = pd.read_csv('insurance.csv')
    return df


df = data_load()


def show_explore_page():
    st.title("Explore the data distribution and interaction")
    st.write(
        """
        ### Data Visualization
        """
    )
    insurance = df
    fig1, axes = plt.subplots(2, 3, figsize=(15, 9))
    bg_color = '#f4f4f4'
    sns.histplot(insurance['age'], color='red', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].patch.set_facecolor(bg_color)

    sns.histplot(insurance['bmi'], color='green', kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('BMI Distribution')
    axes[0, 1].patch.set_facecolor(bg_color)

    sns.histplot(insurance['charges'], color='blue', kde=True, ax=axes[0, 2])

    axes[0, 2].set_title('Charge Distribution')
    axes[0, 2].patch.set_facecolor(bg_color)

    sns.countplot(x='smoker', data=insurance, hue='sex',
                  palette='Set2', ax=axes[1, 0])
    axes[1, 0].set_title('Smoker vs Gender')
    axes[1, 0].patch.set_facecolor(bg_color)

    sns.countplot(x=insurance['region'],
                  hue=insurance['region'], palette='Set1', ax=axes[1, 1])
    axes[1, 1].set_title('Region Distribution')
    axes[1, 1].patch.set_facecolor(bg_color)

    sns.countplot(x=insurance['children'], hue=insurance['children'],
                  legend=False, palette='Set2', ax=axes[1, 2])
    axes[1, 2].set_title('Children Distribution')
    axes[1, 2].patch.set_facecolor(bg_color)

    plt.gcf().patch.set_facecolor(bg_color)

    plt.tight_layout()

    st.pyplot(fig1)

    st.write(
        """
            ### Relationship of the features and target  
            #### Smoking Habit and BMI effect on Charges
            """
    )
    fig2, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    g = sns.stripplot(data=insurance, x='smoker', y='charges', hue='smoker', palette=[
        'blue', 'orange'], legend=True, ax=axes[0])
    g.set_yticklabels(['0k', '10k', '20k', '30k', '40k', '50k', '60k', '65k'])

    axes[1].scatter(insurance.loc[insurance.smoker == 'yes'].bmi,
                    insurance.loc[insurance.smoker == 'yes'].charges, label="yes", marker='o',
                    s=60, edgecolors='black', c='orange'
                    )
    axes[1].set_yticklabels(
        ['0k', '10k', '20k', '30k', '40k', '50k', '60k', '65k'])

    axes[1].scatter(insurance.loc[insurance.smoker == 'no'].bmi,
                    insurance.loc[insurance.smoker == 'no'].charges, label="no", marker='v',
                    s=60, edgecolors='black', c='lightblue'
                    )
    axes[1].set_yticklabels(
        ['0k', '10k', '20k', '30k', '40k', '50k', '60k', '65k'])

    axes[1].set_xlabel('bmi')
    axes[1].set_ylabel('charges')
    axes[1].legend()
    for ax in axes:
        ax.set_facecolor('#f4f4f4')
    plt.gcf().patch.set_facecolor('#f4f4f4')
    st.pyplot(fig2)
    st.write(
        """
            #### Regional Location and Age effect on Charges
            """
    )

    fig3, axes = plt.subplots(1, 2, figsize=(9.5, 4))

    g1 = sns.stripplot(x='region', y='charges', data=insurance, ax=axes[0])
    g1.set_xticklabels(['SW', 'SE', 'NW', 'NE'])
    g1.set_yticklabels(['0k', '10k', '20k', '30k', '40k', '50k', '60k', '65k'])
    g2 = sns.scatterplot(x='age', y='charges',
                         data=insurance, hue='smoker', ax=axes[1])
    g2.set_yticklabels(['0k', '10k', '20k', '30k', '40k', '50k', '60k', '65k'])
    for ax in axes:
        ax.set_facecolor('#f4f4f4')
    plt.gcf().patch.set_facecolor('#f4f4f4')

    st.pyplot(fig3)
    st.write(
        """
            #### Number of Children and Gender effect on Charges
            """
    )

    fig4, axes = plt.subplots(1, 2, figsize=(9, 4))

    g1 = sns.stripplot(x='children', y='charges', data=insurance,
                       hue='children', palette='Set1', ax=axes[0])
    g1.set_yticklabels(['0k', '10k', '20k', '30k', '40k', '50k', '60k', '65k'])
    g1.set_facecolor('#f4f4f4')
    g2 = sns.boxplot(x='sex', y='charges', data=insurance,
                     hue='sex', palette='Set2', ax=axes[1])
    g2.set_yticklabels(['0k', '10k', '20k', '30k', '40k', '50k', '60k', '65k'])
    g2.set_facecolor('#f4f4f4')

    plt.gcf().patch.set_facecolor('#f4f4f4')
    st.pyplot(fig4)

    st.write(
        """
            #### Combined effect of all features on charges
            """
    )
    plt.figure(figsize=(12, 6))
    g = sns.FacetGrid(insurance, col='smoker', row='sex',
                      hue='region', margin_titles=True, height=2.4, aspect=1.5)
    g.map(sns.scatterplot, 'age', 'charges')

    g.add_legend()

    st.pyplot(g)
