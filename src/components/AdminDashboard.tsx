import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Users, Database, UserPlus, Shield, Search, X, Phone, AlertCircle } from 'lucide-react';
import { getAllUsers, createUserManually, updateUserRole, UserProfile } from '../lib/supabase';

const AdminDashboard: React.FC = () => {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTab, setSelectedTab] = useState<'users' | 'database'>('users');
  const [phoneError, setPhoneError] = useState<string | null>(null);

  // New user form state
  const [newUser, setNewUser] = useState({
    email: '',
    password: '',
    display_name: '',
    phone_number: '',
    provider_type: '',
    is_admin: false
  });

  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      const data = await getAllUsers();
      setUsers(data);
    } catch (error) {
      console.error('Error loading users:', error);
    } finally {
      setLoading(false);
    }
  };

  const validatePhoneNumber = (phone: string): boolean => {
    const phoneRegex = /^\+[1-9]\d{1,14}$/;
    return phoneRegex.test(phone);
  };

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setPhoneError(null);

      if (!newUser.phone_number) {
        setPhoneError('Phone number is required');
        return;
      }

      if (!validatePhoneNumber(newUser.phone_number)) {
        setPhoneError('Please enter a valid phone number with country code (e.g., +1234567890)');
        return;
      }

      await createUserManually(newUser.email, newUser.password, {
        display_name: newUser.display_name,
        phone_number: newUser.phone_number,
        provider_type: newUser.provider_type,
        is_admin: newUser.is_admin
      });
      setShowCreateUser(false);
      loadUsers();
    } catch (error) {
      console.error('Error creating user:', error);
      setPhoneError('Failed to create user. Please try again.');
    }
  };

  const handleToggleAdmin = async (userId: string, currentIsAdmin: boolean) => {
    try {
      await updateUserRole(userId, !currentIsAdmin);
      loadUsers();
    } catch (error) {
      console.error('Error updating user role:', error);
    }
  };

  const filteredUsers = users.filter(user => 
    user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.display_name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="bg-dark-surface border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-display font-bold neon-purple-text">Admin Dashboard</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSelectedTab('users')}
            className={`flex items-center px-4 py-2 rounded-lg transition-all ${
              selectedTab === 'users' ? 'bg-neon-purple text-white' : 'bg-gray-800 text-gray-300'
            }`}
          >
            <Users className="mr-2" size={20} />
            Users
          </button>
          <button
            onClick={() => setSelectedTab('database')}
            className={`flex items-center px-4 py-2 rounded-lg transition-all ${
              selectedTab === 'database' ? 'bg-neon-purple text-white' : 'bg-gray-800 text-gray-300'
            }`}
          >
            <Database className="mr-2" size={20} />
            Database
          </button>
        </div>

        {selectedTab === 'users' && (
          <>
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
              <div className="relative flex-1 w-full sm:w-auto">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                <input
                  type="text"
                  placeholder="Search users..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-gray-800 rounded-lg focus:outline-none focus:ring-2 focus:ring-neon-purple"
                />
              </div>
              <button
                onClick={() => setShowCreateUser(true)}
                className="flex items-center px-4 py-2 bg-neon-purple rounded-lg hover:bg-purple-600 transition-all"
              >
                <UserPlus className="mr-2" size={20} />
                Add User
              </button>
            </div>

            <div className="bg-dark-surface rounded-lg border border-gray-800 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-800">
                      <th className="px-6 py-3 text-left">User</th>
                      <th className="px-6 py-3 text-left">Email</th>
                      <th className="px-6 py-3 text-left">Phone</th>
                      <th className="px-6 py-3 text-left">Provider</th>
                      <th className="px-6 py-3 text-left">Role</th>
                      <th className="px-6 py-3 text-left">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredUsers.map((user) => (
                      <tr key={user.id} className="border-t border-gray-800">
                        <td className="px-6 py-4">
                          <div className="flex items-center">
                            {user.avatar_url ? (
                              <img
                                src={user.avatar_url}
                                alt={user.display_name || 'User'}
                                className="w-8 h-8 rounded-full mr-3"
                              />
                            ) : (
                              <div className="w-8 h-8 rounded-full bg-gray-700 mr-3 flex items-center justify-center">
                                <Users size={16} />
                              </div>
                            )}
                            <span>{user.display_name || 'Unnamed User'}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4">{user.email}</td>
                        <td className="px-6 py-4">{user.phone_number || '-'}</td>
                        <td className="px-6 py-4">{user.provider_type || 'email'}</td>
                        <td className="px-6 py-4">
                          <span className={`px-2 py-1 rounded-full text-sm ${
                            user.is_admin ? 'bg-purple-900 text-purple-200' : 'bg-gray-800 text-gray-300'
                          }`}>
                            {user.is_admin ? 'Admin' : 'User'}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <button
                            onClick={() => handleToggleAdmin(user.id, user.is_admin || false)}
                            className="text-neon-purple hover:text-purple-400 transition-colors"
                          >
                            <Shield size={20} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {selectedTab === 'database' && (
          <div className="bg-dark-surface rounded-lg border border-gray-800 p-6">
            <h2 className="text-xl font-semibold mb-4">Database Management</h2>
            <p className="text-gray-400">
              Database management features will be implemented here.
            </p>
          </div>
        )}

        {showCreateUser && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-dark-surface rounded-lg p-6 max-w-md w-full"
            >
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">Create New User</h2>
                <button
                  onClick={() => setShowCreateUser(false)}
                  className="text-gray-400 hover:text-white"
                >
                  <X size={20} />
                </button>
              </div>

              <form onSubmit={handleCreateUser} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Email</label>
                  <input
                    type="email"
                    required
                    value={newUser.email}
                    onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                    className="input-dark w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Password</label>
                  <input
                    type="password"
                    required
                    value={newUser.password}
                    onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                    className="input-dark w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Display Name</label>
                  <input
                    type="text"
                    value={newUser.display_name}
                    onChange={(e) => setNewUser({ ...newUser, display_name: e.target.value })}
                    className="input-dark w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Phone Number *</label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                    <input
                      type="tel"
                      value={newUser.phone_number}
                      onChange={(e) => setNewUser({ ...newUser, phone_number: e.target.value })}
                      className="input-dark w-full pl-10"
                      placeholder="+1234567890"
                      required
                    />
                  </div>
                  {phoneError && (
                    <div className="mt-1 text-sm text-red-400 flex items-center gap-1">
                      <AlertCircle size={14} />
                      <span>{phoneError}</span>
                    </div>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Provider Type</label>
                  <select
                    value={newUser.provider_type}
                    onChange={(e) => setNewUser({ ...newUser, provider_type: e.target.value })}
                    className="input-dark w-full"
                  >
                    <option value="">Select provider</option>
                    <option value="email">Email</option>
                    <option value="google">Google</option>
                    <option value="facebook">Facebook</option>
                    <option value="twitter">Twitter</option>
                  </select>
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="isAdmin"
                    checked={newUser.is_admin}
                    onChange={(e) => setNewUser({ ...newUser, is_admin: e.target.checked })}
                    className="mr-2"
                  />
                  <label htmlFor="isAdmin" className="text-sm font-medium">
                    Make this user an admin
                  </label>
                </div>

                <div className="flex justify-end gap-3">
                  <button
                    type="button"
                    onClick={() => setShowCreateUser(false)}
                    className="px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-neon-purple rounded-lg hover:bg-purple-600 transition-colors"
                  >
                    Create User
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </main>
    </div>
  );
};

export default AdminDashboard;