extern "C" {
#include "cec17.h"
}
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <iomanip>

using namespace std;


template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

// template<typename Iter>
// Iter select_randomly(Iter start, Iter end, int seed) {
//     static std::mt19937 gen(seed);
//     return select_randomly(start, end, gen);
// }


class Location{

    public:
        vector<double> pos;
        double fitness;

        Location(){}

        Location(int dimension){
            pos = vector<double>(dimension);
        }

        double& operator[](int index) {
            return pos[index];
        }

        const double& operator[](int index) const {
            return pos[index];
        }

        void clip(double lower_bound, double upper_bound) {
            for(double& elem: pos){
                elem  = max(lower_bound, min(upper_bound, elem));
            }
        }

        bool operator<=(const Location& other) const {
            return fitness <= other.fitness;
        }

        bool operator<(const Location& other) const {
            return fitness < other.fitness;
        }

        Location operator+(const Location& other) const {
            Location result(pos.size());
            for(int i = 0; i < pos.size(); ++i) {
                result.pos[i] = pos[i] + other.pos[i];
            }
            return result;
        }

        Location operator+=(const Location& other) {
            for(int i = 0; i < pos.size(); ++i) {
                pos[i] += other.pos[i];
            }
            return *this;
        }

        Location operator-(const Location& other) const {
            Location result(pos.size());
            for(int i = 0; i < pos.size(); ++i) {
                result.pos[i] = pos[i] - other.pos[i];
            }
            return result;
        }

        Location operator-=(const Location& other) {
            for(int i = 0; i < pos.size(); ++i) {
                pos[i] -= other.pos[i];
            }
            return *this;
        }

        Location operator*(double scalar) const {
            Location result(pos.size());
            for(int i = 0; i < pos.size(); ++i) {
                result.pos[i] = pos[i] * scalar;
            }
            return result;
        }

        Location operator/(double scalar) const {
            Location result(pos.size());
            for(int i = 0; i < pos.size(); ++i) {
                result.pos[i] = pos[i] / scalar;
            }
            return result;
        }

        friend ostream& operator<<(ostream& os, const Location& location) {
            os << "Position: [";
            for (const auto& value : location.pos) {
                os << fixed << setprecision(3) << setw(7) << value << ", ";
            }
            os << "] ";
            os << "Fitness: " << scientific << setprecision(3) << location.fitness;
            return os;
        }
};


/**
 * @brief Calcula la distancia euclidiana entre dos localizaciones
 * @pre Ambas localizaciones deben definirse en el mismo número de dimensiones 
 * @return Distancia entre las localizaciones (double)
 */
double distance(const Location & a, const Location & b)
{
    // Calculamos el vector que une las localizaciones
    const Location c = b - a;

    // Calculamos el módulo del vector 
    double aux = 0;
    for(const auto& elem : c.pos)
        aux += pow(elem, 2);
    return sqrt(aux);
}

// template<typename RandomGenerator>
// double levy_flight(RandomGenerator& gen)
// {   
//     uniform_real_distribution<> dis(0.0, 1.0);

//     double beta = 1.5;
//     double sigma = 0.696575;

//     double r_a = dis(gen);
//     double r_b = dis(gen);
    
//     // double result = (0.01 * r_a * sigma) / (pow(r_b, 1/beta));
//     // cout << "AA" << result << endl;
//     // return result;
//     return 0.01 * (dis(gen)*sigma) / (pow(dis(gen), 1/beta));
// }


class SquirrelSearchAlgorithm{

    private:
        // Límites inferior y superior de cada dimensión en el espacio de búsqueda
        const double lower_bound = -100.0;
        const double upper_bound =  100.0;

        // Constantes que controlan el comportamiento de las ardillas durante la búsqueda
        const double prob_predation = 0.1;
        const double gliding_constant = 1.9;

        // Vector que almacena las localizaciones de las ardillas de la población 
        vector<Location> locations;

        // Puntero a función que representa la función de evaluación de fitness 
        double (*eval_fitness)(double * sol);

        // Vector que almacena los índices correspondientes a:
        // · Árboles de nueces (Hickory nut tree) 
        vector<int> index_ht;
        // · Árboles de bellotas (Acorn nut tree)
        vector<int> index_at;
        // · Árboles normales (Normal tree)
        vector<int> index_nt;

        // Variable que indica el número máximo de evaluaciones y el contador de evaluaciones 
        int max_evaluations; 
        int cont_evaluations;

        // Variable que indica el número máximo de iteraciones y el contador de iteraciones
        int max_iterations; 
        int cont_iteration;

        // Generador de números aleatorios
        mt19937 & rd_gen;


    public:

        /**
         * @brief Constructor de la clase SquirrelSearchAlgorithm (SSA)
         * 
         * @param population_size Número de ardillas en la población
         * @param dimension_size Número de dimensiones en las que se evalua una localización
         * @param func_fitness Función fitness evaluadora de localizaciones
         * @param max_evaluaciones Número máximo de evaluaciones 
         */
        SquirrelSearchAlgorithm(int population_size, int dimension_size, double (*func_fitness)(double * sol), int max_evaluations,
                                mt19937 & gen)
        : eval_fitness(func_fitness), rd_gen(gen)
        {   
            // Declaramos el generador de números aleatorios obtenidos de una distribución uniforme entre lower_bound y upper_bound
            uniform_real_distribution<> dis(lower_bound, upper_bound);

            // Definimos el vector de localizaciones con el número de localizaciones y de dimensiones indicado
            locations = vector<Location>(population_size, Location(dimension_size));

            // Inicialización aleatoria de localizaciones, dispersas en el espacio de soluciones  
            for(auto& location : locations)
                for(double& elem : location.pos)
                    elem = dis(gen);


            // Se calcula el máximo de iteraciones aprox. como el número máximo de evaluaciones entre el tamaño de la población
            this->max_iterations = max_evaluations / population_size;

            // Se pone a 1 el contador de iteraciones
            cont_iteration = 1;

            //
            this->max_evaluations = max_evaluations; 

            // Se pone a 0 el contador de evaluaciones
            cont_evaluations = 0;

            // Ordena las ubicaciones de la ardilla en la población en base al fitness (orden descendente)
            this->sortByFitness(true);

            // Inicializa los vectores de los índices de árboles de nueces, de bellotas y normales
            index_ht = {0};
            index_at = {1, 2, 3};
            index_nt = vector<int>(46);
            iota(index_nt.begin(), index_nt.end(), 0);
        }

        /**
         * @brief 
         * 
         */
        bool runIteration()
        {
            if (cont_evaluations >= max_evaluations) return false;

            // Declaramos el generador de números aleatorios obtenidos de una distribución uniforme entre lower_bound y upper_bound
            uniform_real_distribution<> dis_lim(lower_bound, upper_bound);

            // Declaramos el generador de números aleatorios obtenidos de una distribución uniforme entre 0 y 1
            uniform_real_distribution<> dis_st(0.0, 1.0);

            // Permutar índices nt (normal tree)
            shuffle(index_nt.begin(), index_nt.end(), rd_gen);

            // Dividimos en dos conjuntos los índices de las ardillas en árboles normales
            vector<int> index_nt_at(index_nt.begin(), index_nt.begin()+index_nt.size() / 2);
            vector<int> index_nt_ht(index_nt.begin()+index_nt.size() / 2, index_nt.end()); 

            // Ardillas de áboles normales que se dirigen hacia árboles con bellotas
            for(int i_nt : index_nt_at)
            {
                if(dis_st(rd_gen) >= prob_predation) {
                    int i_at = *select_randomly(index_at.begin(), index_at.end(), rd_gen);
                    locations[i_nt] += (locations[i_at]-locations[i_nt]) * glidingDistance() * gliding_constant;
                    locations[i_nt].clip(lower_bound, upper_bound);
                }
                else {
                    for(double& elem : locations[i_nt].pos)
                        elem = dis_lim(rd_gen);
                }
            }

            // Ardillas de árboles normales que se dirigen hacia árboles con nueces
            for(int i_nt : index_nt_ht)
            {
                if(dis_st(rd_gen) >= prob_predation) {
                    int i_ht = *select_randomly(index_ht.begin(), index_ht.end(), rd_gen);
                    locations[i_nt] += (locations[i_ht]-locations[i_nt]) * glidingDistance() * gliding_constant;
                    locations[i_nt].clip(lower_bound, upper_bound);
                }
                else {
                    for(double& elem : locations[i_nt].pos)
                        elem = dis_lim(rd_gen);
                }
            }

            // Ardillas de árboles con bellotas que se dirigen hacia árboles con nueces
            for(int i_at : index_at)
            {
                if(dis_st(rd_gen) >= prob_predation) {
                    int i_ht = *select_randomly(index_ht.begin(), index_ht.end(), rd_gen);
                    locations[i_at] += (locations[i_ht]-locations[i_at]) * glidingDistance() * gliding_constant;
                    locations[i_at].clip(lower_bound, upper_bound);
                }
                else {
                    for(double& elem : locations[i_at].pos)
                        elem = dis_lim(rd_gen);
                }
            }

            // Se calcula el valor umbral de la constante estacional
            double S_min = 1e-5 / pow(365, cont_iteration/(max_iterations/2.5));

            // Se calcula la constante estacional
            double S_c = 0;
            int n = 0;
            for(int i_ht: index_ht) {
                for(int i_at: index_at) {
                    S_c += distance(locations[i_ht], locations[i_at]);
                    n++;
                }
            }
            S_c /= n;

            // Si la constante estacional es menor que el umbral, llega el invierno
            if (S_c < S_min) {
                for(int i_nt : index_nt_at) {
                    for(double& elem: locations[i_nt].pos)
                        elem = dis_lim(rd_gen);
                }
            }

            // Se evalúan las localizaciones y se ordenan de mejor a peor
            sortByFitness(false);

            // Se aumenta en 1 el contador de iteraciones
            ++cont_iteration;

            // Se devuelve true ya que no se ha alcanzado el máximo de evaluaciones
            return true;
        }


        /**
         * @brief Devuelve la localización con mayor fitness encontrado hasta el momento
         */
        Location& getBestLocation() {
            return locations.front();
        }

        /**
         * @brief Ordena la población de ardillas por el valor fitness de las ardillas en orden descendente
         */
        void sortByFitness(bool start)
        {
            int jump = (start ? 0 : index_ht.size());

            for(auto it = locations.begin()+jump; it != locations.end(); ++it){
                if(cont_evaluations < max_evaluations) {
                    (*it).fitness = eval_fitness(&(*it).pos[0]);
                    cont_evaluations++;
                }
            }

            sort(locations.begin(), locations.end(),
                [](const auto& a, const auto& b) { return a.fitness < b.fitness; }
            );
        }

        /**
         * @brief Calcula la distancia de planeo
         */
        double glidingDistance() 
        {
            // Declaramos el generador de números aleatorios obtenidos de una distribución uniforme entre 0 y 1
            uniform_real_distribution<> dis(0.675, 1.5);

            // L = 1/2 * ρ * V² * S * C_L
            // donde ρ = 1.204 kg/m³
            //       V = 5.25 m/s
            //       S = 154 cm² = 0.0154 m² 
            //       C_L = U(0.675, 1.5)
            // double L = 1 / (2 * 1.204 * 27.5625 * 0.0154 * dis(gen));
            double L = 0.9783724 / dis(rd_gen);

            // D = 1/2 * ρ * V² * S * C_D
            // donde ρ = 1.204 kg/m³
            //       V = 5.25 m/s
            //       S = 154 cm² = 0.0154 m² 
            //       C_D = 0.6
            // double D = 1 / (2 * 1.204 * 27.5625 * 0.0154 * 0.6);
            double D = 1.6306207;

            // phi = arctan(D/L)
            double phi = atan2(D, L);

            // d_g = h_g / (tan(phi) * sf)
            // donde h_g = 8 m
            //       sf = 18
            return (8.0 / (tan(phi)) * 18.0); 
        }

        /**
         * @brief Muestra las localizaciones de las ardillas voladoras
         * Recorre todas las ubicaciones y muestra las coordenadas y valor fitness de cada una
         */
        void showFlyinSquirrels() 
        {
            cout << endl;
            for(auto& loc : locations){
                cout << loc << endl;
            }
            cout << endl;
        }

};




int main() {
    int dim = 10;
    int seed = 42;
    mt19937 gen(seed); 

    for (int funcid = 1; funcid <= 30; funcid++) {

        cec17_init("SSA_cpp", funcid, dim);

        SquirrelSearchAlgorithm ssa(50, 10, cec17_fitness, 10000*dim, gen);

        while(ssa.runIteration()){}

        double bestFitness = ssa.getBestLocation().fitness;

        cout <<"Best SSA[F" << funcid <<"]: " << scientific << cec17_error(bestFitness) <<endl;
    }
}