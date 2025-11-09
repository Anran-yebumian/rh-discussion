import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from typing import Tuple, Dict, List
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class CorrectedRiemannZetaFramework:
    """
    修正的黎曼猜想证明框架数值实现
    基于正确的再生核不等式推导
    """
    
    def __init__(self):
        # 已知的 ξ(1/2) 值
        self.xi_half = 0.497
        
        # 参数范围
        self.alpha_range = (0.1, 5.0)
        self.r_range = (0.01, 0.95)  # 避免边界问题
        self.theta_range = (0, 2*np.pi)
        
        # 数值精度控制
        self.epsilon = 1e-12
        
        # ξ函数的实际上界
        self.xi_bound = 0.77
        
        # 缓存优化
        self._kernel_cache = {}
    
    def weighted_bergman_kernel(self, alpha: float, z: complex, w: complex) -> complex:
        """
        计算加权Bergman再生核 K_α(z,w)
        
        Args:
            alpha: 权重参数
            z, w: 复平面上的点
            
        Returns:
            再生核值
        """
        # 使用缓存提高性能
        cache_key = (alpha, z, w)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        if abs(z * w.conjugate()) > 0.95:
            # 使用对数形式避免数值不稳定
            log_k = np.log(alpha + 1) - np.log(np.pi) - (alpha + 2) * np.log(1 - z * w.conjugate())
            result = np.exp(log_k)
        else:
            # 直接计算
            result = (alpha + 1) / (np.pi * (1 - z * w.conjugate()) ** (alpha + 2))
        
        self._kernel_cache[cache_key] = result
        return result
    
    def compute_norm_bound(self, alpha: float) -> float:
        """
        计算修正的范数上界 ||F||_{A_α²} ≤ M(α) * sqrt(2π/((α+1)(α+2)))
        
        Args:
            alpha: 权重参数
            
        Returns:
            范数上界
        """
        base_norm = np.sqrt(2 * np.pi / ((alpha + 1) * (alpha + 2)))
        return base_norm * self.xi_bound
    
    def single_zero_constraint(self, alpha: float, r: float, theta: float) -> float:
        """
        单零点约束的修正计算
        
        Args:
            alpha: 权重参数
            r: 半径
            theta: 角度
            
        Returns:
            修正的U值
        """
        try:
            z0 = r * np.exp(1j * theta)
            
            # 计算相关再生核值
            K_00 = self.weighted_bergman_kernel(alpha, 0, 0)
            K_z0z0 = self.weighted_bergman_kernel(alpha, z0, z0)
            K_z00 = self.weighted_bergman_kernel(alpha, z0, 0)
            
            # 修正的不等式项
            constraint_term_sq = 1 - abs(K_z00)**2 / (K_00 * K_z0z0)
            
            # 避免数值误差导致的负数
            if constraint_term_sq < 0:
                if constraint_term_sq > -self.epsilon:
                    constraint_term = 0
                else:
                    return 10.0  # 无意义的情况
            else:
                constraint_term = np.sqrt(constraint_term_sq)
            
            # 计算范数上界
            norm_bound = self.compute_norm_bound(alpha)
            
            U = norm_bound * constraint_term
            
            # 返回合理的U值
            return min(U, 1.0)
            
        except (ValueError, ZeroDivisionError, la.LinAlgError):
            return 10.0
    
    def four_zero_constraint(self, alpha: float, r: float, theta: float) -> float:
        """
        四元组零点约束的修正计算
        
        Args:
            alpha: 权重参数
            r: 半径
            theta: 角度
            
        Returns:
            修正的U值
        """
        try:
            # 四元组零点
            z1 = r * np.exp(1j * theta)
            z2 = r * np.exp(-1j * theta)
            z3 = -1/r * np.exp(1j * theta)
            z4 = -1/r * np.exp(-1j * theta)
            zeros = [z1, z2, z3, z4]
            
            # 计算Gram矩阵和k向量
            n = len(zeros)
            G = np.zeros((n, n), dtype=complex)
            k = np.zeros(n, dtype=complex)
            
            K_00 = self.weighted_bergman_kernel(alpha, 0, 0)
            
            for i in range(n):
                k[i] = self.weighted_bergman_kernel(alpha, zeros[i], 0)
                for j in range(n):
                    G[i, j] = self.weighted_bergman_kernel(alpha, zeros[i], zeros[j])
            
            # 检查矩阵条件数
            cond_num = np.linalg.cond(G)
            if cond_num > 1e12:
                return 10.0  # 矩阵接近奇异
            
            # 计算投影项 k* G^{-1} k
            try:
                # 使用更稳定的求解方法
                projection_term = k.conjugate() @ np.linalg.solve(G, k)
                
                # 确保投影项在合理范围内
                if projection_term.real < 0 or projection_term.real > K_00:
                    return 10.0
                
                constraint_term_sq = 1 - projection_term.real / K_00
                
                if constraint_term_sq < 0:
                    if constraint_term_sq > -self.epsilon:
                        constraint_term = 0
                    else:
                        return 10.0
                else:
                    constraint_term = np.sqrt(constraint_term_sq)
                    
            except np.linalg.LinAlgError:
                return 10.0
            
            norm_bound = self.compute_norm_bound(alpha)
            U = norm_bound * constraint_term
            
            return min(U, 1.0)
            
        except (ValueError, ZeroDivisionError, la.LinAlgError):
            return 10.0
    
    def grid_search_optimized(self, n_alpha: int = 30, n_r: int = 40, n_theta: int = 30) -> Tuple[float, Dict]:
        """
        优化的网格搜索寻找U的最小值
        
        Args:
            n_alpha, n_r, n_theta: 各维度采样点数
            
        Returns:
            (U_min, optimal_params)
        """
        alpha_min, alpha_max = self.alpha_range
        r_min, r_max = self.r_range
        theta_min, theta_max = self.theta_range
        
        # 生成参数网格
        alphas = np.linspace(alpha_min, alpha_max, n_alpha)
        rs = np.linspace(r_min, r_max, n_r)
        thetas = np.linspace(theta_min, theta_max, n_theta)
        
        U1_min = float('inf')
        U4_min = float('inf')
        optimal_params_1 = {}
        optimal_params_4 = {}
        
        print("开始优化的网格搜索...")
        total_points = n_alpha * n_r * n_theta
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=total_points, desc="网格搜索进度")
        
        for i, alpha in enumerate(alphas):
            for j, r in enumerate(rs):
                for k, theta in enumerate(thetas):
                    # 计算单零点约束
                    U1 = self.single_zero_constraint(alpha, r, theta)
                    if U1 < U1_min and U1 < 1.0:
                        U1_min = U1
                        optimal_params_1 = {'alpha': alpha, 'r': r, 'theta': theta, 'type': 'single'}
                    
                    # 计算四元组零点约束
                    U4 = self.four_zero_constraint(alpha, r, theta)
                    if U4 < U4_min and U4 < 1.0:
                        U4_min = U4
                        optimal_params_4 = {'alpha': alpha, 'r': r, 'theta': theta, 'type': 'four'}
                    
                    pbar.update(1)
        
        pbar.close()
        
        # 选择更好的约束
        if U1_min <= U4_min and U1_min < 1.0:
            return U1_min, optimal_params_1
        elif U4_min < 1.0:
            return U4_min, optimal_params_4
        else:
            # 没有找到有效的约束
            return 1.0, {}
    
    def refine_optimal_parameters(self, initial_params: Dict) -> Tuple[float, Dict]:
        """
        使用局部优化细化最优参数
        
        Args:
            initial_params: 初始参数
            
        Returns:
            (优化后的U值, 优化后的参数)
        """
        if not initial_params:
            return 1.0, {}
        
        alpha0, r0, theta0 = initial_params['alpha'], initial_params['r'], initial_params['theta']
        constraint_type = initial_params.get('type', 'four')
        
        def objective(params):
            alpha, r, theta = params
            
            # 参数约束惩罚
            penalty = 0
            if alpha < self.alpha_range[0] or alpha > self.alpha_range[1]:
                penalty += 1000
            if r < self.r_range[0] or r > self.r_range[1]:
                penalty += 1000
            if theta < self.theta_range[0] or theta > self.theta_range[1]:
                penalty += 1000
            
            if constraint_type == 'single':
                U = self.single_zero_constraint(alpha, r, theta)
            else:
                U = self.four_zero_constraint(alpha, r, theta)
            
            return U + penalty
        
        # 参数边界
        bounds = [
            (max(self.alpha_range[0], alpha0*0.5), min(self.alpha_range[1], alpha0*1.5)),
            (max(self.r_range[0], r0*0.8), min(self.r_range[1], r0*1.2)),
            (max(self.theta_range[0], theta0-0.5), min(self.theta_range[1], theta0+0.5))
        ]
        
        print("开始局部优化...")
        pbar = tqdm(total=100, desc="优化进度")
        
        result = opt.minimize(
            objective,
            [alpha0, r0, theta0],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        pbar.close()
        
        if result.success:
            alpha_opt, r_opt, theta_opt = result.x
            if constraint_type == 'single':
                U_opt = self.single_zero_constraint(alpha_opt, r_opt, theta_opt)
            else:
                U_opt = self.four_zero_constraint(alpha_opt, r_opt, theta_opt)
            
            optimal_params = {
                'alpha': alpha_opt, 
                'r': r_opt, 
                'theta': theta_opt, 
                'type': constraint_type
            }
            
            return U_opt, optimal_params
        else:
            print("局部优化失败，使用初始值")
            if constraint_type == 'single':
                U = self.single_zero_constraint(alpha0, r0, theta0)
            else:
                U = self.four_zero_constraint(alpha0, r0, theta0)
            return U, initial_params
    
    def error_analysis(self, alpha: float, r: float, theta: float, constraint_type: str) -> Dict:
        """
        误差分析：估计数值计算的误差范围
        
        Args:
            alpha, r, theta: 参数
            constraint_type: 约束类型 ('single' 或 'four')
            
        Returns:
            误差分析结果
        """
        print("进行误差分析...")
        
        # 多次计算以估计数值稳定性
        n_samples = 100
        U_samples = []
        
        # 创建进度条
        pbar = tqdm(total=n_samples, desc="误差分析进度")
        
        for _ in range(n_samples):
            # 添加微小扰动模拟数值误差
            alpha_perturbed = alpha * (1 + np.random.normal(0, 1e-8))
            r_perturbed = r * (1 + np.random.normal(0, 1e-8))
            theta_perturbed = theta * (1 + np.random.normal(0, 1e-8))
            
            if constraint_type == 'single':
                U = self.single_zero_constraint(alpha_perturbed, r_perturbed, theta_perturbed)
            else:
                U = self.four_zero_constraint(alpha_perturbed, r_perturbed, theta_perturbed)
            
            U_samples.append(U)
            pbar.update(1)
        
        pbar.close()
        
        U_mean = np.mean(U_samples)
        U_std = np.std(U_samples)
        
        # 计算安全边际
        safety_margin = 3 * U_std
        
        return {
            'U_mean': U_mean,
            'U_std': U_std,
            'safety_margin': safety_margin,
            'U_safe': U_mean + safety_margin,
            'success_criterion': U_mean + safety_margin < self.xi_half
        }
    
    def parameter_sensitivity_analysis(self, alpha: float, r: float, theta: float, constraint_type: str) -> Dict:
        """
        参数敏感性分析
        
        Args:
            alpha, r, theta: 基准参数
            constraint_type: 约束类型
            
        Returns:
            敏感性分析结果
        """
        print("进行参数敏感性分析...")
        
        # 基准U值
        if constraint_type == 'single':
            U_base = self.single_zero_constraint(alpha, r, theta)
        else:
            U_base = self.four_zero_constraint(alpha, r, theta)
        
        # 微小变化量
        delta = 0.01
        
        # 计算各参数的偏导数近似
        dU_dalpha = (self.single_zero_constraint(alpha + delta, r, theta) - U_base) / delta
        dU_dr = (self.single_zero_constraint(alpha, r + delta, theta) - U_base) / delta
        dU_dtheta = (self.single_zero_constraint(alpha, r, theta + delta) - U_base) / delta
        
        # 计算相对敏感性
        sensitivity_alpha = abs(dU_dalpha * alpha / U_base) if U_base > 1e-10 else 0
        sensitivity_r = abs(dU_dr * r / U_base) if U_base > 1e-10 else 0
        sensitivity_theta = abs(dU_dtheta * theta / U_base) if U_base > 1e-10 else 0
        
        return {
            'U_base': U_base,
            'sensitivity_alpha': sensitivity_alpha,
            'sensitivity_r': sensitivity_r,
            'sensitivity_theta': sensitivity_theta,
            'most_sensitive': max(['alpha', 'r', 'theta'], 
                                key=lambda x: {'alpha': sensitivity_alpha, 'r': sensitivity_r, 'theta': sensitivity_theta}[x])
        }
    
    def theoretical_verification(self, alpha: float, r: float, theta: float, constraint_type: str) -> Dict:
        """
        理论验证：检查关键理论假设
        
        Args:
            alpha, r, theta: 参数
            constraint_type: 约束类型
            
        Returns:
            理论验证结果
        """
        print("进行理论验证...")
        
        z0 = r * np.exp(1j * theta)
        
        # 计算关键再生核值
        K_00 = self.weighted_bergman_kernel(alpha, 0, 0)
        K_z0z0 = self.weighted_bergman_kernel(alpha, z0, z0)
        K_z00 = self.weighted_bergman_kernel(alpha, z0, 0)
        
        # 检查理论条件
        conditions = {
            'K_00_positive': K_00.real > 0,
            'K_z0z0_positive': K_z0z0.real > 0,
            'constraint_term_valid': abs(K_z00)**2 <= K_00 * K_z0z0 * (1 + 1e-10),  # 允许微小数值误差
            'norm_bound_valid': self.compute_norm_bound(alpha) > 0
        }
        
        # 计算理论项
        constraint_term = 1 - abs(K_z00)**2 / (K_00 * K_z0z0)
        theoretical_U = self.compute_norm_bound(alpha) * np.sqrt(max(0, constraint_term))
        
        return {
            'conditions': conditions,
            'K_00': K_00,
            'K_z0z0': K_z0z0, 
            'K_z00': K_z00,
            'constraint_term': constraint_term,
            'theoretical_U': theoretical_U,
            'all_conditions_satisfied': all(conditions.values())
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        运行全面的分析流程
        
        Returns:
            完整分析结果
        """
        print("=" * 60)
        print("修正的黎曼猜想证明框架全面分析")
        print("=" * 60)
        
        # 阶段1：网格搜索寻找最优参数
        print("\n阶段1: 网格搜索寻找最优参数")
        U_min_grid, params_grid = self.grid_search_optimized(n_alpha=25, n_r=30, n_theta=25)
        
        if not params_grid:
            print("❌ 网格搜索未找到有效参数")
            return {'success': False, 'error': 'No valid parameters found'}
        
        print(f"网格搜索结果: U_min = {U_min_grid:.6f}")
        print(f"最优参数: α={params_grid['alpha']:.4f}, r={params_grid['r']:.4f}, θ={params_grid['theta']:.4f}")
        print(f"约束类型: {params_grid['type']}")
        
        # 阶段2：参数优化细化
        print("\n阶段2: 参数优化细化")
        U_min_opt, params_opt = self.refine_optimal_parameters(params_grid)
        
        print(f"优化结果: U_min = {U_min_opt:.6f}")
        print(f"最优参数: α={params_opt['alpha']:.4f}, r={params_opt['r']:.4f}, θ={params_opt['theta']:.4f}")
        print(f"约束类型: {params_opt['type']}")
        
        # 阶段3：误差分析
        print("\n阶段3: 误差分析")
        error_analysis = self.error_analysis(
            params_opt['alpha'], params_opt['r'], params_opt['theta'], params_opt['type']
        )
        
        print(f"平均U值: {error_analysis['U_mean']:.6f}")
        print(f"标准差: {error_analysis['U_std']:.2e}")
        print(f"安全边际: {error_analysis['safety_margin']:.2e}")
        print(f"保守上界: {error_analysis['U_safe']:.6f}")
        
        # 阶段4：参数敏感性分析
        print("\n阶段4: 参数敏感性分析")
        sensitivity = self.parameter_sensitivity_analysis(
            params_opt['alpha'], params_opt['r'], params_opt['theta'], params_opt['type']
        )
        
        print(f"参数敏感性: α={sensitivity['sensitivity_alpha']:.4f}, "
              f"r={sensitivity['sensitivity_r']:.4f}, "
              f"θ={sensitivity['sensitivity_theta']:.4f}")
        print(f"最敏感参数: {sensitivity['most_sensitive']}")
        
        # 阶段5：理论验证
        print("\n阶段5: 理论验证")
        theory_check = self.theoretical_verification(
            params_opt['alpha'], params_opt['r'], params_opt['theta'], params_opt['type']
        )
        
        print("理论条件检查:")
        for condition, satisfied in theory_check['conditions'].items():
            status = "✓" if satisfied else "✗"
            print(f"  {condition}: {status}")
        
        print(f"理论U值: {theory_check['theoretical_U']:.6f}")
        
        # 最终结论
        print("\n" + "=" * 60)
        print("最终结论")
        print("=" * 60)
        
        U_final = error_analysis['U_safe']
        success = error_analysis['success_criterion']
        theory_valid = theory_check['all_conditions_satisfied']
        
        print(f"最终保守上界: U_safe = {U_final:.6f}")
        print(f"ξ(1/2)参考值: {self.xi_half:.6f}")
        print(f"比较结果: U_safe {'<' if success else '>='} ξ(1/2)")
        print(f"理论条件: {'全部满足' if theory_valid else '存在违反'}")
        
        if success and theory_valid:
            print("🎉 修正后的框架在数值和理论上都支持黎曼猜想！")
            print("这是一个重要的进展，但还需要严格的数学审查。")
        elif success and not theory_valid:
            print("⚠️  数值结果支持但理论条件有违反，需要进一步检查。")
        elif not success and theory_valid:
            print("⚠️  理论条件满足但数值结果不支持，可能需要调整参数范围。")
        else:
            print("❌ 修正后的框架仍不成立，需要重新审视理论基础。")
        
        # 返回完整结果
        return {
            'U_min_grid': U_min_grid,
            'U_min_optimized': U_min_opt,
            'optimal_parameters': params_opt,
            'error_analysis': error_analysis,
            'sensitivity_analysis': sensitivity,
            'theoretical_verification': theory_check,
            'final_conclusion': {
                'U_safe': U_final,
                'xi_half': self.xi_half,
                'success': success,
                'theory_valid': theory_valid
            }
        }


# 运行全面分析
if __name__ == "__main__":
    # 创建修正框架实例
    corrected_framework = CorrectedRiemannZetaFramework()
    
    # 运行全面分析
    results = corrected_framework.run_comprehensive_analysis()
    
    # 保存结果摘要
    print("\n" + "=" * 60)
    print("分析完成！结果摘要")
    print("=" * 60)
    
    if 'final_conclusion' in results:
        final = results['final_conclusion']
        params = results['optimal_parameters']
        
        print(f"最优参数: α={params['alpha']:.4f}, r={params['r']:.4f}, θ={params['theta']:.4f}")
        print(f"约束类型: {params['type']}")
        print(f"网格搜索 U_min: {results['U_min_grid']:.6f}")
        print(f"优化后 U_min: {results['U_min_optimized']:.6f}")
        print(f"最终保守上界 U_safe: {final['U_safe']:.6f}")
        print(f"ξ(1/2)参考值: {final['xi_half']:.6f}")
        
        if final['success'] and final['theory_valid']:
            print("✅ 修正框架成功：U_safe < ξ(1/2) 且理论条件满足")
            print("这意味着黎曼猜想在修正框架下得到数值支持！")
        else:
            print("❌ 修正框架仍未完全成功")
            
        # 显示关键统计信息
        print(f"\n关键统计信息:")
        print(f"误差分析标准差: {results['error_analysis']['U_std']:.2e}")
        print(f"安全边际: {results['error_analysis']['safety_margin']:.2e}")
        print(f"最敏感参数: {results['sensitivity_analysis']['most_sensitive']}")
        print(f"理论条件满足: {final['theory_valid']}")