from wordcloud import WordCloud, STOPWORDS
from os import  path
import  matplotlib
matplotlib.use('TkAgg')
def wordcloudify(frequencies, fakeness):



    def grey_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "rgb({}, 0, {})".format(min(int(255 * fakeness[word]),255), max(0,int(255 * (1 - fakeness[word]))))

    d = path.dirname(__file__)

    # preprocessing the text a little bit
    text = frequencies
    print(text)

    # adding movie script specific stopwords

    wc = WordCloud(max_words=len(frequencies), margin=10,
                   random_state=1, width=2000, height=1400, background_color='white', relative_scaling=0.001) \
        .generate_from_frequencies(text)
    # store default colored image
    default_colors = wc.to_array()
    matplotlib.pyplot.title("Custom colors")
    matplotlib.pyplot.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
                             interpolation="bilinear")
    wc.to_file("a_new_hope.png")
    matplotlib.pyplot.axis("off")
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title("Default colors")
    matplotlib.pyplot.imshow(default_colors, interpolation="bilinear")
    matplotlib.pyplot.axis("off")
    matplotlib.pyplot.show()
    x = 1
