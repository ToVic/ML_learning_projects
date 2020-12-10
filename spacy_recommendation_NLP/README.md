A miniproject done within the subject 4IZ481 at FIS VSE

With only a little time I had to come up with a good utilization of NLP for a certain business case. In the few hours I had I created a fast solution, a very simple notebook.
We were given the famous Kaggle hotel review data. I had to cope with the characteristics of the data as it contained not only positive, but also negative reviews.
Eventually I came up with code which analyzes the writer's style and recommends hotels in desired destination according to his writing style, represented by proportions 
of word types used and word count. Division of the styles was done with kmeans clustering algorithm, which mathematically does not make much sense given the data, but
it's just a quick solution with a ton of room for improvement, looking especially at the superslow function for vectorization.

The notebook can be easily converted into a working py module and a working miniapp.
