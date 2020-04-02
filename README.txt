---- ECM2423 Coursework exercise read-me file ----

There are 4 python files

    - 8_puzzle.py

        to run this please call the main() method
        the entire program is generic, can be run for any N value, and the user can enter their own puzzle for each N config
        as well as use test puzzles for N = 3 and N = 4 puzzles (8, 15 puzzles)

    - k_means_clustering.py

        to run this please call the main((int) k1, (int) k2) method
        for example - main(10, 20) means starting at 10 clusters and ending at 20 clusters
        where k1 is the number of clusters (starting cluster amount)
        and k2 is the number of clusters (end cluster amount) - this is for the displaying the accuracy from k1 to k2

    - my_own_k_means_clustering.py (my own k-means implementation)

        to run this please call the main((arr) data-set, (int) number of clusters) method
        I have a method that loads the digits data-set and returns the correct data-set format
        This variable is called "training" and my k-means clustering works for any value of k (number of clusters)

    - pokemon_k_means_clustering.py

        to run this please call the main() method
        this will display a scatter diagram of the clusters formed from the pokemon attack against defense

