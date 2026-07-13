#pragma once
#include <fstream>
#include <random>
#include <vector>
#include <filesystem>
#include <iostream>
#include "multicrit.h"

class Instance {
    public:

    static std::vector<double> speedByHour;     // average vehicle speed depending on in-day hour
    static std::vector<int> populations;        // town populations
    static std::vector<int> cityLocations;      // town number of locations
    static std::vector<int> cityDepots;         // town number of depots

    int n;      // number of delivery orders
    int m;      // number of parcel locker locations
    int v;      // number of vehicles

    int P;      // park and depart time (seconds)
    int S;      // service time (seconds)
    int C;      // vehicle capacity (kilograms)

    int B;      // starting time

    std::vector<std::vector<int>> dist;         // distance between location pairs in meters
    std::vector<std::vector<int>> lockers;      // number of available lockers of each size for each locker location
                                                //      locker of a given size can contain order of the same size or smaller

    std::vector<int> weights;                   // order weight (in kilograms)
    std::vector<int> sizes;                     // order size (0, 1 or 2)
    std::vector<int> locations;                 // order location
    std::vector<int> deliveries;                // is order a delivery (true) or pickup (false)

    /*
        seed
        city = city number (starting at 0)
        orderFactor = number of delivery orders relative to city population (how many orders per 1 citizen)
        pickupToDeliveryFactor = number of pickup orders relative to number of delivery order (0 = only delivery orders, 1 = only pickup orders)
        vehicleFactor = number of vehicles relative to the number of orders (number of vehicles per order)
        occupancyProbability = chance for a locker to be unavailable (either for delivery or pickup), from o to 1
        parkDeparkTime = time (in seconds) needed to park/depark
        serviceTime = time (in seconds) needed to service (deliver/pickup) order
        startingTime = time (in seconds) when the delivery day starts
    */
    static Instance * createRandom(long seed, int city, double orderFactor, double pickupToDeliveryFactor, double locationFactor, double vehicleFactor, double occupancyProbability, int parkDepartTime, int serviceTime, int vehicleCapacity, int startingTime) {
        Instance * inst = new Instance();

        std::mt19937 mtg(seed);
        std::uniform_real_distribution<double> urd(0.0, 1.0);
        std::uniform_int_distribution<int> uidSizes(0, 2);

        std::vector<double> weightThresholds;
        double threshold = 0.0;
        double factor = 1.1;
        double currentFactor = 1.0;
        for (int i = 0; i < 25; ++i) {
            threshold += currentFactor;
            weightThresholds.push_back(threshold);
            currentFactor /= factor;
        }

        std::uniform_real_distribution<double> urdWeights(0.0, weightThresholds[24]);

        inst->S = serviceTime;
        inst->P = parkDepartTime;
        inst->C = vehicleCapacity;
        inst->B = startingTime;

        inst->n = populations[city] * orderFactor + 0.5;
        
        int noPickupOrders = inst->n * pickupToDeliveryFactor + 0.5;
        int noDeliveryOrders = inst->n - noPickupOrders;

        inst->n = noDeliveryOrders + noPickupOrders;

        int depots = cityDepots[city];
        int lockerLocations = cityLocations[city];
        int fileLines = lockerLocations + depots;
        inst->m = locationFactor * lockerLocations + 0.5;

        char str[20];
        sprintf(str, "matrix/%1d.csv", city);
        std::ifstream ifs(str);

        std::vector<std::vector<int>> readDistances;

        double raw;
        for (int k = 0; k < fileLines; ++k) {
            readDistances.push_back(std::vector<int>());
            for (int l = 0; l < fileLines; ++l) {
                ifs >> raw;
                readDistances[k].push_back(raw + 0.5);
            }
        }

        std::uniform_int_distribution<int> uidDepots(lockerLocations, lockerLocations + depots - 1);
        int depotLine = uidDepots(mtg);

        for (int k = 0; k <= inst->m; ++k) {
            inst->dist.push_back(std::vector<int>());
            for (int l = 0; l <= inst->m; ++l) {
                int i;
                int j;

                if (k == 0)
                    i = depotLine;
                else
                    i = k - 1;
                if (l == 0)
                    j = depotLine;
                else
                    j = l - 1;

                inst->dist[k].push_back(readDistances[i][j]);
            }
        }

        std::uniform_int_distribution<int> uidLocations(1, inst->m);

        inst->v = populations[city] * orderFactor * vehicleFactor + 0.5;
        if (inst->v < 1)
            inst->v = 1;

        int maxA = 32;
        int maxB = 29;
        int maxC = 18;
        for (int k = 0; k <= inst->m; ++k) {
            inst->lockers.push_back(std::vector<int>());

            if (k == 0)
                continue; 
            for (int i = 0; i < 3; ++i)
                inst->lockers[k].push_back(0);
            for (int i = 0; i < maxA; ++i)
                if (urd(mtg) >= occupancyProbability)
                    inst->lockers[k][0]++;
            for (int i = 0; i < maxB; ++i)
                if (urd(mtg) >= occupancyProbability)
                    inst->lockers[k][1]++;
            for (int i = 0; i < maxC; ++i)
                if (urd(mtg) >= occupancyProbability)
                    inst->lockers[k][2]++;

        }

        std::vector<int> hist;
        for (int j = 0; j < 25; ++j)
            hist.push_back(0);

        for (int i = 0; i < inst->n; ++i) {
            int size = uidSizes(mtg);
            int weight;
            double draw = urdWeights(mtg);
            for (int j = 0; j < 25; ++j) {
                if (draw < weightThresholds[j]) {
                    weight = j + 1;
                    hist[j]++;
                    break;
                }
            }

            int location;
            int delivery;

            if (i < noDeliveryOrders) {
                location = uidLocations(mtg);
                delivery = true;
            }
            else {  
                bool okay = false;
                while (! okay) {
                    location = uidLocations(mtg);
                    for (int j = size; j < 3; ++j) {
                        if (inst->lockers[location][j] > 0) {
                            inst->lockers[location][j]--;
                            size = j;
                            okay = true;
                            break;
                        }
                    }
                }
                delivery = false;
            }

            inst->sizes.push_back(size);
            inst->weights.push_back(weight);
            inst->locations.push_back(location);
            inst->deliveries.push_back(delivery);
        }

        return inst;
    };

    void printShort() {
        printf("n=%d m=%d v=%d S=%d P=%d C=%d B=%.2lf\n", n, m, v, S, P, C, B / 3600.0);
    }

    void print() {
        printf("n=%d m=%d v=%d S=%d P=%d C=%d B=%.2lf\n", n, m, v, S, P, C, B / 3600.0);
        for (int i = 0; i < n; ++i)
            printf("%d s=%d w=%-2d l=%-3d d=%d\n", i, sizes[i], weights[i], locations[i], deliveries[i]);
        for (int k = 0; k <= m; ++k) {
            for (int l = 0; l <= m; ++l)
                printf("%5d ", dist[k][l]);
            printf("\n");
        }
        for (int k = 1; k <= m; ++k) {
            printf("%d %d %d %d\n", k, lockers[k][0], lockers[k][1], lockers[k][2]);
                
        }
    }

    void printToFile(std::string str) {
        FILE * fptr = fopen(str.c_str(), "w");
        fprintf(fptr, "%d %d %d\n", n, m, v);
        fprintf(fptr, "%d %d %d %.2lf\n", S, P, C, B / 3600.0);
        for (int i = 0; i < n; ++i)
            fprintf(fptr, "%d %-2d %-3d %d\n", sizes[i], weights[i], locations[i], deliveries[i]);
        for (int k = 0; k <= m; ++k) {
            for (int l = 0; l <= m; ++l)
                fprintf(fptr, "%5d ", dist[k][l]);
            fprintf(fptr, "\n");
        }
        for (int k = 1; k <= m; ++k) {
            fprintf(fptr, "%d %d %d %d\n", k, lockers[k][0], lockers[k][1], lockers[k][2]);
                
        }
        fclose(fptr); 
    }    

    // travel time between locations in seconds
    inline int getTravelTime(int from, int to, int time) {
        return (double) (dist[from][to]) / speedByHour[(time % 86400) / 3600] + 0.5;
    }

    static void init() {
        createSpeedByHour();
        createPopulations();
        createDepotNumbers();
        createLocationNumbers();
    }

    static void createPopulations() {
        populations.clear();
        populations.push_back(1794166);  // 0 = Warszawa
        populations.push_back(641928);   // 1 = Wrocław
        populations.push_back(672185);   // 2 = Łódź
        populations.push_back(217530);   // 3 = Częstochowa
        populations.push_back(209296);   // 4 = Radom
        populations.push_back(71674);    // 5 = Inowrocław
        populations.push_back(71560);    // 6 = Ostrów Wielkopolski
        populations.push_back(69639);    // 7 = Suwałki
        populations.push_back(26421);    // 8 = Kłodzko
        populations.push_back(26114);    // 9 = Biłgoraj
    }

    static void createDepotNumbers() {
        cityDepots.clear();
        cityDepots.push_back(6);  // 0 = Warszawa
        cityDepots.push_back(3);  // 1 = Wrocław
        cityDepots.push_back(2);  // 2 = Łódź
        cityDepots.push_back(1);  // 3 = Częstochowa
        cityDepots.push_back(1);  // 4 = Radom
        cityDepots.push_back(1);  // 5 = Inowrocław
        cityDepots.push_back(1);  // 6 = Ostrów Wielkopolski
        cityDepots.push_back(1);  // 7 = Suwałki
        cityDepots.push_back(1);  // 8 = Kłodzko
        cityDepots.push_back(1);  // 9 = Biłgoraj
    }

    static void createLocationNumbers() {
        cityLocations.clear();
        cityLocations.push_back(949);   // 0 = Warszawa
        cityLocations.push_back(375);   // 1 = Wrocław
        cityLocations.push_back(300);   // 2 = Łódź
        cityLocations.push_back(39);    // 3 = Częstochowa
        cityLocations.push_back(81);    // 4 = Radom
        cityLocations.push_back(21);    // 5 = Inowrocław
        cityLocations.push_back(46);    // 6 = Ostrów Wielkopolski
        cityLocations.push_back(29);    // 7 = Suwałki
        cityLocations.push_back(11);    // 8 = Kłodzko
        cityLocations.push_back(13);    // 9 = Biłgoraj
    }       

    static void createSpeedByHour() {
        speedByHour.clear();
        speedByHour.push_back(38.9 / 3.6);
        speedByHour.push_back(39.5 / 3.6);
        speedByHour.push_back(40.2 / 3.6);
        speedByHour.push_back(40.9 / 3.6);
        speedByHour.push_back(41.0 / 3.6);
        speedByHour.push_back(40.0 / 3.6);
        speedByHour.push_back(35.6 / 3.6);
        speedByHour.push_back(30.9 / 3.6);
        speedByHour.push_back(30.2 / 3.6);
        speedByHour.push_back(30.8 / 3.6);
        speedByHour.push_back(31.1 / 3.6);
        speedByHour.push_back(31.7 / 3.6);
        speedByHour.push_back(32.4 / 3.6);
        speedByHour.push_back(32.1 / 3.6);
        speedByHour.push_back(31.2 / 3.6);
        speedByHour.push_back(30.9 / 3.6);
        speedByHour.push_back(30.2 / 3.6);
        speedByHour.push_back(28.4 / 3.6);
        speedByHour.push_back(28.4 / 3.6);
        speedByHour.push_back(31.1 / 3.6);
        speedByHour.push_back(32.5 / 3.6);
        speedByHour.push_back(33.6 / 3.6);
        speedByHour.push_back(37.0 / 3.6);
        speedByHour.push_back(38.0 / 3.6);
    }    
};

std::vector<double> Instance::speedByHour = std::vector<double>();
std::vector<int> Instance::populations = std::vector<int>();
std::vector<int> Instance::cityLocations = std::vector<int>();
std::vector<int> Instance::cityDepots = std::vector<int>();
